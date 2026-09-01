import { lookAt, normalize, cross, add, scale, sub } from './mat.js';

/**
 * Which key does what. Layout-independent: `event.code` reports the physical
 * key, so W is the key where W sits on a QWERTY board whatever the user's
 * layout says it types. Binding by `event.key` instead would scatter the
 * controls across the keyboard on AZERTY and Dvorak.
 */
export const MOVE_KEYS = {
  KeyW: 'forward', ArrowUp: 'forward',
  KeyS: 'back', ArrowDown: 'back',
  KeyA: 'left', ArrowLeft: 'left',
  KeyD: 'right', ArrowRight: 'right',
  KeyE: 'up', Space: 'up',
  KeyQ: 'down', KeyC: 'down',
};

/** Orbit camera with mouse, wheel, touch, and keyboard fly-through. */
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
    /** Movement directions currently held down. */
    this.held = new Set();
    this.boost = false;
    /**
     * Travel per second, as a fraction of the viewing distance.
     *
     * Tied to distance rather than fixed in world units because a splat scene
     * has no inherent scale: a tabletop capture and a street are both "one
     * scene", and a speed that suits either is unusable in the other. Zooming
     * out to see more also speeds up crossing it, which is what one expects.
     */
    this.moveSpeed = 1.1;
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
    let moved = this.applyMovement(dtSeconds);
    if (this.autoRotate && !this._pointers.size) {
      this.yaw += this.autoRotateSpeed * dtSeconds;
      moved = true;
    }
    if (moved) this.onChange();
    return moved;
  }

  /**
   * Fly the camera by whatever movement keys are held.
   *
   * Scaled by elapsed time rather than counted per event, so travel is the
   * same on a 30 fps laptop as on a 144 Hz monitor -- and so holding a key
   * accelerates nothing when the tab stutters.
   *
   * Moving the orbit target is what carries the camera: it keeps its distance
   * and angles, so dragging still orbits whatever you flew to rather than
   * snapping back to where you started.
   */
  applyMovement(dtSeconds) {
    if (!this.held.size) return false;
    // A stalled tab can hand back a huge delta; clamping keeps a held key from
    // teleporting the camera across the scene on the first frame back.
    const dt = Math.min(Math.max(dtSeconds, 0), 0.1);
    if (dt <= 0) return false;

    const forward = normalize(sub(this.target, this.position));
    const right = normalize(cross(forward, [0, 1, 0]));
    const step = this.distance * this.moveSpeed * dt * (this.boost ? 3 : 1);

    let delta = [0, 0, 0];
    if (this.held.has('forward')) delta = add(delta, forward);
    if (this.held.has('back')) delta = sub(delta, forward);
    if (this.held.has('right')) delta = add(delta, right);
    if (this.held.has('left')) delta = sub(delta, right);
    if (this.held.has('up')) delta = add(delta, [0, 1, 0]);
    if (this.held.has('down')) delta = sub(delta, [0, 1, 0]);

    const length = Math.hypot(delta[0], delta[1], delta[2]);
    // Opposite keys cancel exactly; diagonals must not travel faster than a
    // straight line, which is what normalising prevents.
    if (length < 1e-6) return false;
    this.target = add(this.target, scale(delta, step / length));
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

    /**
     * Does this key event belong to someone typing?
     *
     * The keys are bound on the window so the viewer does not have to be
     * focused to fly, which means every text field on the page shares them.
     * Without this check, typing "would" into the notes field flies the camera
     * across the scene and drops half the letters.
     */
    const isTyping = (e) => {
      const t = e.target;
      if (!t || t === el) return false;
      if (t.isContentEditable) return true;
      const tag = t.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    };

    const keyDown = (e) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      const direction = MOVE_KEYS[e.code];
      if (!direction) return;
      // Space and the arrows scroll the page otherwise, which fights the
      // movement they are meant to drive.
      e.preventDefault();
      this.held.add(direction);
      this.boost = e.shiftKey;
    };
    const keyUp = (e) => {
      const direction = MOVE_KEYS[e.code];
      if (direction) this.held.delete(direction);
      this.boost = e.shiftKey;
    };
    /**
     * Release everything when focus leaves.
     *
     * A key held as the window loses focus never delivers its keyup, so the
     * camera would fly off in that direction forever and no amount of clicking
     * back would stop it -- the key is not down any more, so there is nothing
     * left to release it.
     */
    const release = () => { this.held.clear(); this.boost = false; };

    const view = el.ownerDocument?.defaultView;
    if (view?.addEventListener) {
      view.addEventListener('keydown', keyDown);
      view.addEventListener('keyup', keyUp);
      view.addEventListener('blur', release);
      el.ownerDocument.addEventListener('visibilitychange', release);
    }

    el.addEventListener('pointerdown', down);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', wheel, { passive: false });
    el.addEventListener('contextmenu', menu);

    return () => {
      if (view?.removeEventListener) {
        view.removeEventListener('keydown', keyDown);
        view.removeEventListener('keyup', keyUp);
        view.removeEventListener('blur', release);
        el.ownerDocument.removeEventListener('visibilitychange', release);
      }
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
