import { VERTEX_SHADER, FRAGMENT_SHADER } from './shaders.js';
import { OrbitCamera } from './camera.js';
import { perspective, eulerToMat3, mat3ToColumnMajor } from './mat.js';

/**
 * WebGL2 gaussian splat viewer.
 *
 * Splat data is uploaded once into textures; every frame draws one instanced
 * quad per splat in the order the sort worker most recently produced. Edits are
 * uniforms, so the editing UI stays at full frame rate on large clouds.
 */

const MAX_TEXTURE_SIDE = 4096;

export class SplatViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    if (!this.gl) throw new Error('WebGL2 is not available in this browser');

    this.count = 0;
    this.sortedCount = 0;
    this.generation = 0;
    this.background = [0.043, 0.051, 0.071];
    this.edits = null;
    this.showCropped = false;
    this.stats = { fps: 0, drawn: 0, sortMs: 0 };

    this._needsSort = true;
    this._lastSortKey = '';
    this._frameTimes = [];
    this._lastFrame = performance.now();
    this._running = false;

    this.camera = new OrbitCamera(canvas, { onChange: () => { this._needsSort = true; } });
    this._initGl();
    this._initWorker();
    this._observeResize();
  }

  _initGl() {
    const gl = this.gl;
    this.program = linkProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    gl.useProgram(this.program);

    this.uniforms = {};
    const count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(this.program, i);
      this.uniforms[info.name] = gl.getUniformLocation(this.program, info.name);
    }

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    // One unit quad, reused by every instance.
    const corners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.cornerBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);
    const cornerLoc = gl.getAttribLocation(this.program, 'aCorner');
    gl.enableVertexAttribArray(cornerLoc);
    gl.vertexAttribPointer(cornerLoc, 2, gl.FLOAT, false, 0, 0);

    // Per-instance splat id, rewritten each time the sorter finishes.
    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    this.indexLoc = gl.getAttribLocation(this.program, 'aSplatIndex');
    gl.enableVertexAttribArray(this.indexLoc);
    gl.vertexAttribIPointer(this.indexLoc, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(this.indexLoc, 1);

    gl.bindVertexArray(null);

    this.geometryTexture = gl.createTexture();
    this.colorTexture = gl.createTexture();

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    // Premultiplied "over", matching the fragment shader's output.
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.blendEquation(gl.FUNC_ADD);
  }

  _initWorker() {
    this.worker = new Worker(new URL('./sortWorker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'sorted') {
        // A sort that started before the latest data load is stale.
        if (msg.generation !== this.generation) return;
        const order = new Uint32Array(msg.order);
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
        this.sortedCount = msg.count;
        this.stats.sortMs = Math.round(performance.now() - this._sortStarted);
      } else if (msg.type === 'error') {
        console.error('splat sorter:', msg.message);
      }
    };
  }

  /**
   * Load a cloud from the compact 32-byte-per-splat `.splat` format.
   * @param {ArrayBuffer} buffer
   */
  load(buffer) {
    const gl = this.gl;
    const count = Math.floor(buffer.byteLength / 32);
    if (!count) throw new Error('this splat file is empty');
    if (count > MAX_TEXTURE_SIDE * MAX_TEXTURE_SIDE / 3) {
      throw new Error(`${count.toLocaleString()} splats exceeds what this device can hold`);
    }

    const view = new DataView(buffer);
    const positions = new Float32Array(count * 3);

    // Geometry texture: 3 RGBA32F texels per splat.
    const geomTexels = count * 3;
    const geomWidth = Math.min(MAX_TEXTURE_SIDE, Math.max(1, geomTexels));
    const geomHeight = Math.ceil(geomTexels / geomWidth);
    const geom = new Float32Array(geomWidth * geomHeight * 4);

    const colorWidth = Math.min(MAX_TEXTURE_SIDE, Math.max(1, count));
    const colorHeight = Math.ceil(count / colorWidth);
    const colors = new Uint8Array(colorWidth * colorHeight * 4);

    for (let i = 0; i < count; i++) {
      const o = i * 32;
      const x = view.getFloat32(o, true);
      const y = view.getFloat32(o + 4, true);
      const z = view.getFloat32(o + 8, true);
      positions[i * 3] = x; positions[i * 3 + 1] = y; positions[i * 3 + 2] = z;

      const g = i * 12;
      geom[g + 0] = x;
      geom[g + 1] = y;
      geom[g + 2] = z;
      geom[g + 3] = view.getUint8(o + 27) / 255;              // opacity

      geom[g + 4] = view.getFloat32(o + 12, true);            // scale
      geom[g + 5] = view.getFloat32(o + 16, true);
      geom[g + 6] = view.getFloat32(o + 20, true);
      geom[g + 7] = 0;

      // Rotation is stored as bytes; normalise here so the shader need not.
      let qw = (view.getUint8(o + 28) - 128) / 128;
      let qx = (view.getUint8(o + 29) - 128) / 128;
      let qy = (view.getUint8(o + 30) - 128) / 128;
      let qz = (view.getUint8(o + 31) - 128) / 128;
      const n = Math.hypot(qw, qx, qy, qz) || 1;
      geom[g + 8] = qw / n; geom[g + 9] = qx / n; geom[g + 10] = qy / n; geom[g + 11] = qz / n;

      colors[i * 4 + 0] = view.getUint8(o + 24);
      colors[i * 4 + 1] = view.getUint8(o + 25);
      colors[i * 4 + 2] = view.getUint8(o + 26);
      colors[i * 4 + 3] = 255;
    }

    gl.bindTexture(gl.TEXTURE_2D, this.geometryTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, geomWidth, geomHeight, 0, gl.RGBA, gl.FLOAT, geom);
    setNearest(gl);

    gl.bindTexture(gl.TEXTURE_2D, this.colorTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, colorWidth, colorHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, colors);
    setNearest(gl);

    this.count = count;
    this.geomWidth = geomWidth;
    this.colorWidth = colorWidth;
    this.generation += 1;
    this.sortedCount = 0;

    // Draw in load order until the first sort lands, so the cloud appears at once.
    const identity = new Uint32Array(count);
    for (let i = 0; i < count; i++) identity[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, identity, gl.DYNAMIC_DRAW);
    this.sortedCount = count;

    const copy = positions.slice();
    this.worker.postMessage({ type: 'load', positions: copy.buffer }, [copy.buffer]);
    this.bounds = boundsOf(positions, count);
    this._needsSort = true;
    return { count, bounds: this.bounds };
  }

  setEdits(edits) {
    this.edits = edits;
    if (edits?.background) this.background = hexToRgb(edits.background);
    this._needsSort = true;
  }

  start() {
    if (this._running) return;
    this._running = true;
    const loop = () => {
      if (!this._running) return;
      this._frame();
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    this._running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  dispose() {
    this.stop();
    this.worker?.terminate();
    this.camera.dispose();
    this._resizeObserver?.disconnect();
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.cornerBuffer);
    gl.deleteBuffer(this.indexBuffer);
    gl.deleteTexture(this.geometryTexture);
    gl.deleteTexture(this.colorTexture);
    gl.deleteVertexArray(this.vao);
  }

  /** PNG data URL of the current view, for library thumbnails and sharing. */
  snapshot() {
    this._frame();
    return this.canvas.toDataURL('image/png');
  }

  /**
   * PNG Blob of the current view. Preferred over `snapshot()` where the result
   * is handed to a download or an upload, since it avoids a base64 round trip
   * (and the `data:` fetch that would otherwise need a CSP exception).
   */
  snapshotBlob() {
    this._frame();
    return new Promise((resolve) => this.canvas.toBlob(resolve, 'image/png'));
  }

  _frame() {
    const gl = this.gl;
    const now = performance.now();
    const dt = (now - this._lastFrame) / 1000;
    this._lastFrame = now;

    this._frameTimes.push(dt);
    if (this._frameTimes.length > 30) this._frameTimes.shift();
    const mean = this._frameTimes.reduce((a, b) => a + b, 0) / this._frameTimes.length;
    this.stats.fps = mean > 0 ? Math.round(1 / mean) : 0;

    this.camera.tick(dt);

    const width = this.canvas.width;
    const height = this.canvas.height;
    gl.viewport(0, 0, width, height);
    gl.clearColor(this.background[0], this.background[1], this.background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.count) return;

    const view = this.camera.viewMatrix();
    const projection = perspective(this.camera.fov, width / height, 0.05, 200);
    const e = this.edits || {};
    const rot = eulerToMat3(...(e.rotate || [0, 0, 0]));
    const editScale = e.scale ?? 1;
    const translate = e.translate || [0, 0, 0];

    this._maybeSort(view, rot, editScale, translate);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.geometryTexture);
    gl.uniform1i(this.uniforms.uGeometry, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTexture);
    gl.uniform1i(this.uniforms.uColor, 1);

    gl.uniform1i(this.uniforms.uGeometryWidth, this.geomWidth);
    gl.uniform1i(this.uniforms.uColorWidth, this.colorWidth);
    gl.uniformMatrix4fv(this.uniforms.uView, false, view);
    gl.uniformMatrix4fv(this.uniforms.uProjection, false, projection);
    gl.uniform2f(this.uniforms.uViewport, width, height);
    gl.uniform1f(this.uniforms.uFocal, height / (2 * Math.tan(this.camera.fov / 2)));

    gl.uniformMatrix3fv(this.uniforms.uEditRotation, false, mat3ToColumnMajor(rot));
    gl.uniform3fv(this.uniforms.uEditTranslate, new Float32Array(translate));
    gl.uniform1f(this.uniforms.uEditScale, editScale);
    gl.uniform1f(this.uniforms.uSplatScale, e.splatScale ?? 1);
    gl.uniform1f(this.uniforms.uOpacity, e.opacity ?? 1);
    gl.uniform1f(this.uniforms.uExposure, e.exposure ?? 0);
    gl.uniform1f(this.uniforms.uSaturation, e.saturation ?? 1);
    gl.uniform1f(this.uniforms.uPruneBelow, e.pruneBelowOpacity ?? 0);

    const crop = e.crop;
    gl.uniform1i(this.uniforms.uCropEnabled, crop ? (crop.invert ? 2 : 1) : 0);
    gl.uniform3fv(this.uniforms.uCropMin, new Float32Array(crop?.min || [0, 0, 0]));
    gl.uniform3fv(this.uniforms.uCropMax, new Float32Array(crop?.max || [0, 0, 0]));
    gl.uniform1i(this.uniforms.uHighlightCropped, this.showCropped ? 1 : 0);

    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.sortedCount);
    this.stats.drawn = this.sortedCount;
    gl.bindVertexArray(null);
  }

  /**
   * Depth for splat p is `(view · (R·p·s + t)).z`, which is affine in p — so the
   * worker only needs three coefficients and a bias, not the matrices.
   */
  _maybeSort(view, rot, editScale, translate) {
    const axis = [
      editScale * (view[2] * rot[0] + view[6] * rot[3] + view[10] * rot[6]),
      editScale * (view[2] * rot[1] + view[6] * rot[4] + view[10] * rot[7]),
      editScale * (view[2] * rot[2] + view[6] * rot[5] + view[10] * rot[8]),
    ];
    const bias = view[2] * translate[0] + view[6] * translate[1] + view[10] * translate[2] + view[14];

    // Re-sort only on a meaningful change; small jitter is invisible and a sort
    // per frame would keep a big cloud permanently busy.
    const key = [...axis, bias].map((v) => v.toFixed(3)).join(',');
    if (key === this._lastSortKey && !this._needsSort) return;
    this._lastSortKey = key;
    this._needsSort = false;
    this._sortStarted = performance.now();
    this.worker.postMessage({ type: 'sort', axis, bias, generation: this.generation });
  }

  _observeResize() {
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
    };
    resize();
    this._resizeObserver = new ResizeObserver(resize);
    this._resizeObserver.observe(this.canvas);
  }
}

function setNearest(gl) {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader failed to compile: ${log}`);
  }
  return shader;
}

function linkProgram(gl, vsSource, fsSource) {
  const program = gl.createProgram();
  const vs = compile(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSource);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`shader program failed to link: ${gl.getProgramInfoLog(program)}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function boundsOf(positions, count) {
  if (!count) return { center: [0, 0, 0], radius: 1, min: [0, 0, 0], max: [0, 0, 0] };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i * 3 + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  const center = [0, 1, 2].map((k) => (min[k] + max[k]) / 2);
  const radius = Math.max(1e-3, Math.hypot(max[0] - center[0], max[1] - center[1], max[2] - center[2]));
  return { min, max, center, radius };
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
