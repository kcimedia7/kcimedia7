import {
  createCloud, sigmoid, logit, shToColor, colorToSh, quatNormalize,
} from './splat.js';

/**
 * Reader/writer for the binary PLY layout that 3D Gaussian Splatting trainers
 * emit (INRIA's `point_cloud.ply` and everything compatible with it):
 *
 *   x y z  nx ny nz  f_dc_0..2  [f_rest_0..N]  opacity  scale_0..2  rot_0..3
 *
 * Higher-order spherical harmonics (`f_rest_*`) are view-dependent colour. The
 * viewer here evaluates degree 0 only, so those are parsed past but not kept —
 * on export we write the degree-0 form, which every viewer can read.
 */

const TYPE_SIZES = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

function readerFor(type) {
  switch (type) {
    case 'char': case 'int8': return (v, o) => v.getInt8(o);
    case 'uchar': case 'uint8': return (v, o) => v.getUint8(o);
    case 'short': case 'int16': return (v, o) => v.getInt16(o, true);
    case 'ushort': case 'uint16': return (v, o) => v.getUint16(o, true);
    case 'int': case 'int32': return (v, o) => v.getInt32(o, true);
    case 'uint': case 'uint32': return (v, o) => v.getUint32(o, true);
    case 'float': case 'float32': return (v, o) => v.getFloat32(o, true);
    case 'double': case 'float64': return (v, o) => v.getFloat64(o, true);
    default: throw new Error(`unsupported PLY scalar type: ${type}`);
  }
}

export function parsePlyHeader(buffer) {
  const probe = buffer.subarray(0, Math.min(buffer.length, 64 * 1024)).toString('latin1');
  const endIdx = probe.indexOf('end_header');
  if (!probe.startsWith('ply') || endIdx === -1) throw new Error('not a PLY file');
  const nl = probe.indexOf('\n', endIdx);
  const headerText = probe.slice(0, nl);
  const dataStart = nl + 1;

  let format = null;
  const elements = [];
  for (const raw of headerText.split(/\r?\n/)) {
    const parts = raw.trim().split(/\s+/);
    if (parts[0] === 'format') format = parts[1];
    else if (parts[0] === 'element') elements.push({ name: parts[1], count: Number(parts[2]), properties: [] });
    else if (parts[0] === 'property' && elements.length) {
      const el = elements[elements.length - 1];
      if (parts[1] === 'list') {
        el.properties.push({ list: true, countType: parts[2], type: parts[3], name: parts[4] });
      } else {
        el.properties.push({ list: false, type: parts[1], name: parts[2] });
      }
    }
  }
  if (!format) throw new Error('PLY header has no format line');
  return { format, elements, dataStart };
}

export function decodePly(buffer) {
  const { format, elements, dataStart } = parsePlyHeader(buffer);
  if (format !== 'binary_little_endian') {
    if (format === 'ascii') return decodeAsciiPly(buffer, elements, dataStart);
    throw new Error(`unsupported PLY format: ${format} (need binary_little_endian or ascii)`);
  }
  const vertex = elements.find((e) => e.name === 'vertex');
  if (!vertex) throw new Error('PLY has no vertex element');
  if (vertex.properties.some((p) => p.list)) throw new Error('list properties are not supported on vertices');

  const offsets = {};
  let stride = 0;
  for (const p of vertex.properties) {
    const size = TYPE_SIZES[p.type];
    if (!size) throw new Error(`unsupported PLY scalar type: ${p.type}`);
    offsets[p.name] = { offset: stride, read: readerFor(p.type) };
    stride += size;
  }

  const needed = ['x', 'y', 'z'];
  for (const n of needed) if (!(n in offsets)) throw new Error(`PLY is missing property "${n}"`);

  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const count = vertex.count;
  if (dataStart + count * stride > buffer.length) throw new Error('PLY body is shorter than its header claims');

  const cloud = createCloud(count);
  const get = (name, i) => {
    const p = offsets[name];
    return p ? p.read(view, dataStart + i * stride + p.offset) : undefined;
  };
  const hasGaussian = 'scale_0' in offsets && 'rot_0' in offsets && 'opacity' in offsets;
  const hasDc = 'f_dc_0' in offsets;
  const hasRgb = 'red' in offsets;

  for (let i = 0; i < count; i++) {
    cloud.positions[i * 3 + 0] = get('x', i);
    cloud.positions[i * 3 + 1] = get('y', i);
    cloud.positions[i * 3 + 2] = get('z', i);

    if (hasGaussian) {
      for (let k = 0; k < 3; k++) cloud.scales[i * 3 + k] = Math.exp(get(`scale_${k}`, i));
      const q = quatNormalize([get('rot_0', i), get('rot_1', i), get('rot_2', i), get('rot_3', i)]);
      for (let k = 0; k < 4; k++) cloud.rotations[i * 4 + k] = q[k];
      cloud.opacities[i] = sigmoid(get('opacity', i));
    } else {
      // A plain point cloud (e.g. COLMAP's sparse output): give every point a
      // small isotropic gaussian so it is still renderable.
      for (let k = 0; k < 3; k++) cloud.scales[i * 3 + k] = 0.01;
      cloud.rotations[i * 4 + 0] = 1;
      cloud.opacities[i] = 1;
    }

    if (hasDc) {
      for (let k = 0; k < 3; k++) {
        cloud.colors[i * 3 + k] = clamp01(shToColor(get(`f_dc_${k}`, i)));
      }
    } else if (hasRgb) {
      cloud.colors[i * 3 + 0] = get('red', i) / 255;
      cloud.colors[i * 3 + 1] = get('green', i) / 255;
      cloud.colors[i * 3 + 2] = get('blue', i) / 255;
    } else {
      cloud.colors[i * 3 + 0] = cloud.colors[i * 3 + 1] = cloud.colors[i * 3 + 2] = 0.5;
    }
  }
  return cloud;
}

function decodeAsciiPly(buffer, elements, dataStart) {
  const vertex = elements.find((e) => e.name === 'vertex');
  if (!vertex) throw new Error('PLY has no vertex element');
  const names = vertex.properties.map((p) => p.name);
  const lines = buffer.subarray(dataStart).toString('utf8').split(/\r?\n/);
  const cloud = createCloud(vertex.count);
  for (let i = 0; i < vertex.count; i++) {
    const nums = lines[i].trim().split(/\s+/).map(Number);
    const row = {};
    names.forEach((n, k) => { row[n] = nums[k]; });
    cloud.positions[i * 3 + 0] = row.x;
    cloud.positions[i * 3 + 1] = row.y;
    cloud.positions[i * 3 + 2] = row.z;
    const gaussian = 'scale_0' in row;
    for (let k = 0; k < 3; k++) cloud.scales[i * 3 + k] = gaussian ? Math.exp(row[`scale_${k}`]) : 0.01;
    if (gaussian) {
      const q = quatNormalize([row.rot_0, row.rot_1, row.rot_2, row.rot_3]);
      for (let k = 0; k < 4; k++) cloud.rotations[i * 4 + k] = q[k];
      cloud.opacities[i] = sigmoid(row.opacity);
    } else {
      cloud.rotations[i * 4] = 1;
      cloud.opacities[i] = 1;
    }
    for (let k = 0; k < 3; k++) {
      cloud.colors[i * 3 + k] = 'f_dc_0' in row
        ? clamp01(shToColor(row[`f_dc_${k}`]))
        : ('red' in row ? row[['red', 'green', 'blue'][k]] / 255 : 0.5);
    }
  }
  return cloud;
}

export function encodePly(cloud) {
  const props = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
  ];
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    'comment generated by SplatWorks',
    `element vertex ${cloud.count}`,
    ...props.map((p) => `property float ${p}`),
    'end_header',
    '',
  ].join('\n');

  const headerBuf = Buffer.from(header, 'latin1');
  const stride = props.length * 4;
  const body = Buffer.allocUnsafe(cloud.count * stride);
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);

  for (let i = 0; i < cloud.count; i++) {
    let o = i * stride;
    const put = (v) => { view.setFloat32(o, v, true); o += 4; };
    put(cloud.positions[i * 3 + 0]);
    put(cloud.positions[i * 3 + 1]);
    put(cloud.positions[i * 3 + 2]);
    put(0); put(0); put(0); // normals are unused by 3DGS but conventionally present
    for (let k = 0; k < 3; k++) put(colorToSh(cloud.colors[i * 3 + k]));
    put(logit(cloud.opacities[i]));
    for (let k = 0; k < 3; k++) put(Math.log(Math.max(cloud.scales[i * 3 + k], 1e-8)));
    const q = quatNormalize([
      cloud.rotations[i * 4 + 0], cloud.rotations[i * 4 + 1],
      cloud.rotations[i * 4 + 2], cloud.rotations[i * 4 + 3],
    ]);
    for (let k = 0; k < 4; k++) put(q[k]);
  }
  return Buffer.concat([headerBuf, body]);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
