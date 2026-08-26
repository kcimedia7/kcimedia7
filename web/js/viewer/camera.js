import { lookAt, normalize, cross, add, scale, sub } from './mat.js';

/** Orbit camera with mouse, wheel, and touch input. */
export class OrbitCamera {
  constructor(canvas, { onChange } = {}) {
    this.canvas = canvas;
    this.onChange = onChange || (() => {});
    this.target = [0, 0, 0];
    this.distance = 3.2;
    this.yaw = 0;
    this.pitch = -0.2;
    this.fov = (55 * Math.PI) / 180;
    this.minDistance = 0.15;
    this.maxDistance = 60;
    this.autoRotate = false;
    this.autoRotateSpeed = 0.25;
    this._pointers = new Map();
    this._pinchDistance = 0;
    this._detach = this._attach();
  }

  get position() {
    const cp = Math.cos(this.pitch);
    return add(this.target, [
      Math.sin(this.yaw) * cp * this.distance,
      Math.sin(this.pitch) * this.distance,
      Math.cos(this.yaw) * cp * this.distance,
    ]);
  }

  viewMatrix() {
    return lookAt(this.position, this.target, [0, 1, 0]);
  }

  frame(bounds, { padding = 1.6 } = {}) {
    if (bounds?.center) this.target = [...bounds.center];
    const radius = bounds?.radius || 1;
    this.distance = clamp((radius * padding) / Math.tan(this.fov / 2), this.minDistance, this.maxDistance);
    this.onChange();
  }

  reset() {
    this.yaw = 0;
    this.pitch = -0.2;
    this.onChange();
  }

  setView(name) {
    const views = {
      front: [0, 0], back: [Math.PI, 0], left: [-Math.PI / 2, 0],
      right: [Math.PI / 2, 0], top: [0, -Math.PI / 2 + 0.001], bottom: [0, Math.PI / 2 - 0.001],
    };
    const v = views[name];
    if (!v) return;
    [this.yaw, this.pitch] = v;
    this.onChange();
  }

  tick(dtSeconds) {
    if (!this.autoRotate || this._pointers.size) return false;
    this.yaw += this.autoRotateSpeed * dtSeconds;
    this.onChange();
    return true;
  }

  orbit(dx, dy) {
    this.yaw -= dx * 0.005;
    this.pitch = clamp(this.pitch - dy * 0.005, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
    this.onChange();
  }

  pan(dx, dy) {
    const forward = normalize(sub(this.target, this.position));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    // Pan by the world distance one pixel covers at the target plane.
    const k = (2 * this.distance * Math.tan(this.fov / 2)) / this.canvas.clientHeight;
    this.target = add(this.target, add(scale(right, -dx * k), scale(up, dy * k)));
    this.onChange();
  }

  zoom(factor) {
    this.distance = clamp(this.distance * factor, this.minDistance, this.maxDistance);
    this.onChange();
  }

  dispose() {
    this._detach();
  }

  _attach() {
    const el = this.canvas;
    const down = (e) => {
      el.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
      if (this._pointers.size === 2) this._pinchDistance = this._touchSpan();
    };
    const move = (e) => {
      const prev = this._pointers.get(e.pointerId);
      if (!prev) return;
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
      this._pointers.set(e.pointerId, { ...prev, x: e.clientX, y: e.clientY });

      if (this._pointers.size >= 2) {
        const span = this._touchSpan();
        if (this._pinchDistance > 0 && span > 0) this.zoom(this._pinchDistance / span);
        this._pinchDistance = span;
        this.pan(dx / 2, dy / 2);
        return;
      }
      const panning = prev.button === 1 || prev.button === 2 || e.shiftKey || prev.shift;
      if (panning) this.pan(dx, dy);
      else this.orbit(dx, dy);
    };
    const up = (e) => {
      this._pointers.delete(e.pointerId);
      if (this._pointers.size < 2) this._pinchDistance = 0;
      if (el.hasPointerCapture?.(e.pointerId)) el.releasePointerCapture(e.pointerId);
    };
    const wheel = (e) => {
      e.preventDefault();
      this.zoom(Math.exp(e.deltaY * 0.0012));
    };
    const menu = (e) => e.preventDefault();

    el.addEventListener('pointerdown', down);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', wheel, { passive: false });
    el.addEventListener('contextmenu', menu);

    return () => {
      el.removeEventListener('pointerdown', down);
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      el.removeEventListener('wheel', wheel);
      el.removeEventListener('contextmenu', menu);
    };
  }

  _touchSpan() {
    const pts = [...this._pointers.values()];
    if (pts.length < 2) return 0;
    return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  }
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
