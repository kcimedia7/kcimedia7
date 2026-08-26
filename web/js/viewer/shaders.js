/**
 * EWA splatting shaders.
 *
 * Each gaussian is drawn as one instanced quad. The vertex shader projects the
 * splat's 3D covariance into a 2D screen-space covariance, eigen-decomposes it
 * to get the ellipse axes, and sizes the quad to 3 sigma along each. The
 * fragment shader evaluates the gaussian and outputs premultiplied alpha, so
 * back-to-front "over" compositing gives the correct result.
 *
 * Splat attributes live in two textures indexed by splat id:
 *   uGeometry  RGBA32F, 3 texels per splat — position+opacity, scale, rotation
 *   uColor     RGBA8,   1 texel per splat  — linear-ish RGB
 *
 * Edits (transform, colour grade, crop, prune) are uniforms applied here, so
 * dragging a slider is free — nothing is re-uploaded.
 */

export const VERTEX_SHADER = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

in vec2 aCorner;        // unit quad corner, -1..1
in uint aSplatIndex;    // depth-sorted splat id

uniform sampler2D uGeometry;
uniform sampler2D uColor;
uniform int uGeometryWidth;
uniform int uColorWidth;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec2 uViewport;
uniform float uFocal;

// Edits
uniform mat3 uEditRotation;
uniform vec3 uEditTranslate;
uniform float uEditScale;
uniform float uSplatScale;
uniform float uOpacity;
uniform float uExposure;
uniform float uSaturation;
uniform vec3 uCropMin;
uniform vec3 uCropMax;
uniform int uCropEnabled;      // 0 off, 1 keep inside, 2 keep outside
uniform float uPruneBelow;
uniform int uHighlightCropped; // draw clipped splats faintly instead of hiding

out vec4 vColor;
out vec2 vGaussian;

vec4 fetchGeometry(uint index, int slot) {
  int i = int(index) * 3 + slot;
  return texelFetch(uGeometry, ivec2(i % uGeometryWidth, i / uGeometryWidth), 0);
}

mat3 quatToMat3(vec4 q) {
  // q is (w, x, y, z), already normalised on upload.
  float w = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y),
    2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y)
  );
}

void cull() {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // outside the clip volume
}

void main() {
  vec4 posOpacity = fetchGeometry(aSplatIndex, 0);
  vec4 scaleTexel = fetchGeometry(aSplatIndex, 1);
  vec4 rotation   = fetchGeometry(aSplatIndex, 2);

  vec3 origin = posOpacity.xyz;
  float alpha = posOpacity.w * uOpacity;

  float clippedFade = 1.0;
  if (uCropEnabled != 0) {
    // Crop is evaluated against untransformed coordinates so the box a user
    // drew stays put when they later rotate or move the model.
    bool inside = all(greaterThanEqual(origin, uCropMin)) && all(lessThanEqual(origin, uCropMax));
    bool keep = (uCropEnabled == 1) ? inside : !inside;
    if (!keep) {
      if (uHighlightCropped == 0) { cull(); return; }
      clippedFade = 0.10;
    }
  }

  if (alpha < uPruneBelow) { cull(); return; }

  vec3 center = uEditRotation * origin * uEditScale + uEditTranslate;

  vec4 cam = uView * vec4(center, 1.0);
  if (cam.z > -0.05) { cull(); return; }  // at or behind the eye

  vec4 clip = uProjection * cam;
  float guard = 1.3 * clip.w;
  if (clip.z < -guard || abs(clip.x) > guard || abs(clip.y) > guard) { cull(); return; }

  // World-space 3D covariance: Sigma = R S S^T R^T.
  vec3 scl = scaleTexel.xyz * uEditScale * uSplatScale;
  mat3 rot = uEditRotation * quatToMat3(rotation);
  mat3 M = rot * mat3(scl.x, 0.0, 0.0, 0.0, scl.y, 0.0, 0.0, 0.0, scl.z);
  mat3 sigma = M * transpose(M);

  // Project: camera-space covariance, then the perspective Jacobian.
  mat3 W = mat3(uView);
  float invZ = 1.0 / cam.z;
  float invZ2 = invZ * invZ;
  mat3 J = mat3(
    -uFocal * invZ, 0.0,            0.0,
    0.0,            -uFocal * invZ, 0.0,
    uFocal * cam.x * invZ2, uFocal * cam.y * invZ2, 0.0
  );
  mat3 T = J * W;
  mat3 cov = T * sigma * transpose(T);

  // Dilate by half a pixel so distant splats stay visible instead of aliasing.
  float a = cov[0][0] + 0.3;
  float b = cov[0][1];
  float c = cov[1][1] + 0.3;

  float mid = 0.5 * (a + c);
  float disc = sqrt(max(mid * mid - (a * c - b * b), 1e-9));
  float lambda1 = mid + disc;
  float lambda2 = mid - disc;
  if (lambda2 <= 0.0) { cull(); return; }

  vec2 axis1 = normalize(abs(b) < 1e-9 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
                                       : vec2(b, lambda1 - a));
  vec2 axis2 = vec2(-axis1.y, axis1.x);

  // Clamp so one huge near-camera splat cannot cost the whole frame in fill.
  float r1 = min(3.0 * sqrt(lambda1), 0.35 * uViewport.x);
  float r2 = min(3.0 * sqrt(lambda2), 0.35 * uViewport.x);

  vec2 offsetPx = aCorner.x * axis1 * r1 + aCorner.y * axis2 * r2;

  vec3 rgb = texelFetch(uColor, ivec2(int(aSplatIndex) % uColorWidth,
                                      int(aSplatIndex) / uColorWidth), 0).rgb;
  rgb *= exp2(uExposure);
  float grey = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
  rgb = clamp(mix(vec3(grey), rgb, uSaturation), 0.0, 1.0);

  vColor = vec4(rgb, clamp(alpha, 0.0, 1.0) * clippedFade);
  vGaussian = aCorner * 3.0;   // distance in sigmas

  gl_Position = vec4(clip.xy / clip.w + offsetPx / uViewport * 2.0, clip.z / clip.w, 1.0);
}
`;

export const FRAGMENT_SHADER = /* glsl */ `#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vGaussian;
out vec4 fragColor;

void main() {
  float d = dot(vGaussian, vGaussian);
  if (d > 9.0) discard;              // beyond 3 sigma
  float alpha = exp(-0.5 * d) * vColor.a;
  if (alpha < 0.004) discard;
  // Premultiplied alpha for back-to-front "over" compositing.
  fragColor = vec4(vColor.rgb * alpha, alpha);
}
`;
