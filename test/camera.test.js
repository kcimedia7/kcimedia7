import test from 'node:test';
import assert from 'node:assert/strict';
import { OrbitCamera, MOVE_KEYS } from '../web/js/viewer/camera.js';

/**
 * Fly-through movement, checked without a browser.
 *
 * The camera only needs an element that accepts listeners and reports a size,
 * so a stub is enough to drive the real event handlers -- including the two
 * that matter most: not stealing keys from text fields, and releasing held
 * keys when focus leaves.
 */
function makeTarget() {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) { listeners.get(type)?.delete(fn); },
    dispatch(type, event = {}) {
      for (const fn of listeners.get(type) || []) fn({ preventDefault() {}, ...event });
    },
    count(type) { return listeners.get(type)?.size || 0; },
  };
}

function makeCamera() {
  const view = makeTarget();
  const document = makeTarget();
  document.defaultView = view;
  const canvas = makeTarget();
  canvas.ownerDocument = document;
  canvas.clientHeight = 600;
  canvas.clientWidth = 800;
  const camera = new OrbitCamera(canvas, { onChange: () => {} });
  return { camera, view, document, canvas };
}

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

test('holding a movement key flies the camera through the scene', () => {
  const { camera } = makeCamera();
  const before = [...camera.target];
  const eye = [...camera.position];

  camera.held.add('forward');
  assert.equal(camera.applyMovement(0.1), true);

  assert.ok(distance(camera.target, before) > 0, 'the camera did not move');
  // The eye travels with the target: distance and angles are unchanged, so a
  // drag still orbits whatever you flew to rather than snapping back.
  assert.ok(Math.abs(distance(camera.position, camera.target) - camera.distance) < 1e-6);
  assert.ok(distance(camera.position, eye) > 0, 'the eye should travel with the target');

  // Forward means towards what the camera is looking at.
  const moved = distance(camera.position, before);
  assert.ok(moved < camera.distance, 'forward should close the gap, not widen it');
});

test('travel depends on elapsed time, not on how many frames were drawn', () => {
  // Otherwise the same key held for the same second crosses a scene at 144 Hz
  // and barely moves at 30.
  const oneStep = makeCamera().camera;
  oneStep.held.add('forward');
  oneStep.applyMovement(0.1);

  const manySteps = makeCamera().camera;
  manySteps.held.add('forward');
  for (let i = 0; i < 10; i++) manySteps.applyMovement(0.01);

  assert.ok(distance(oneStep.target, manySteps.target) < 1e-6,
    `one 0.1s step and ten 0.01s steps must agree: ${oneStep.target} vs ${manySteps.target}`);
});

test('a stalled tab cannot teleport the camera on the frame it resumes', () => {
  // requestAnimationFrame hands back a huge delta after the tab was hidden.
  const { camera } = makeCamera();
  camera.held.add('forward');
  const start = [...camera.target];
  camera.applyMovement(30);
  const travelled = distance(camera.target, start);
  const oneTick = camera.distance * camera.moveSpeed * 0.1;
  assert.ok(travelled <= oneTick + 1e-6,
    `a 30 second delta moved ${travelled}, more than the ${oneTick} cap`);
});

test('opposite keys cancel and diagonals are not faster than straight lines', () => {
  const { camera } = makeCamera();
  camera.held.add('forward');
  camera.held.add('back');
  const before = [...camera.target];
  assert.equal(camera.applyMovement(0.1), false, 'opposed keys should not move at all');
  assert.deepEqual(camera.target, before);

  const straight = makeCamera().camera;
  straight.held.add('forward');
  straight.applyMovement(0.1);
  const straightDistance = distance(straight.target, [0, 0, 0]);

  const diagonal = makeCamera().camera;
  diagonal.held.add('forward');
  diagonal.held.add('right');
  diagonal.applyMovement(0.1);
  const diagonalDistance = distance(diagonal.target, [0, 0, 0]);

  assert.ok(Math.abs(diagonalDistance - straightDistance) < 1e-6,
    `diagonal travelled ${diagonalDistance} against ${straightDistance} straight`);
});

test('speed follows the viewing distance so any size of scene is crossable', () => {
  // A tabletop capture and a street are both "one scene"; a fixed speed in
  // world units is unusable in one or the other.
  const near = makeCamera().camera;
  near.distance = 1;
  near.held.add('forward');
  near.applyMovement(0.1);
  const nearMove = distance(near.target, [0, 0, 0]);

  const far = makeCamera().camera;
  far.distance = 100;
  far.held.add('forward');
  far.applyMovement(0.1);
  const farMove = distance(far.target, [0, 0, 0]);

  assert.ok(Math.abs(farMove / nearMove - 100) < 0.01,
    `speed should scale with distance, got a ratio of ${farMove / nearMove}`);
});

test('shift moves faster without changing direction', () => {
  const plain = makeCamera().camera;
  plain.held.add('forward');
  plain.applyMovement(0.1);

  const boosted = makeCamera().camera;
  boosted.held.add('forward');
  boosted.boost = true;
  boosted.applyMovement(0.1);

  const a = distance(plain.target, [0, 0, 0]);
  const b = distance(boosted.target, [0, 0, 0]);
  assert.ok(b > a * 2, `boost should be clearly faster: ${a} vs ${b}`);
});

test('typing in a text field does not fly the camera', () => {
  // The keys are bound on the window so the viewer need not be focused, which
  // means every input on the page shares them. Typing "would" into the notes
  // field must not launch the camera across the scene.
  const { camera, view } = makeCamera();
  for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT']) {
    view.dispatch('keydown', { code: 'KeyW', target: { tagName } });
    assert.equal(camera.held.size, 0, `${tagName} should keep its own keystrokes`);
  }
  view.dispatch('keydown', { code: 'KeyW', target: { tagName: 'DIV', isContentEditable: true } });
  assert.equal(camera.held.size, 0, 'contenteditable should keep its own keystrokes');

  // But the same key over the page at large does move.
  view.dispatch('keydown', { code: 'KeyW', target: { tagName: 'BODY' } });
  assert.deepEqual([...camera.held], ['forward']);
});

test('a shortcut with a modifier is left to the browser', () => {
  const { camera, view } = makeCamera();
  for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) {
    view.dispatch('keydown', { code: 'KeyD', target: { tagName: 'BODY' }, [modifier]: true });
    assert.equal(camera.held.size, 0, `${modifier}+D belongs to the browser`);
  }
});

test('losing focus releases every held key', () => {
  // A key held as the window loses focus never delivers its keyup, so without
  // this the camera flies off forever and clicking back cannot stop it.
  const { camera, view, document } = makeCamera();
  view.dispatch('keydown', { code: 'KeyW', target: { tagName: 'BODY' }, shiftKey: true });
  view.dispatch('keydown', { code: 'KeyD', target: { tagName: 'BODY' } });
  assert.equal(camera.held.size, 2);

  view.dispatch('blur');
  assert.equal(camera.held.size, 0, 'blur must release everything');
  assert.equal(camera.boost, false);

  view.dispatch('keydown', { code: 'KeyW', target: { tagName: 'BODY' } });
  document.dispatch('visibilitychange');
  assert.equal(camera.held.size, 0, 'switching tabs must release everything too');
});

test('keys are bound by physical position, not by the letter they type', () => {
  // event.code is layout-independent. Binding event.key instead scatters the
  // controls across the keyboard on AZERTY and Dvorak.
  assert.equal(MOVE_KEYS.KeyW, 'forward');
  assert.equal(MOVE_KEYS.KeyA, 'left');
  assert.equal(MOVE_KEYS.KeyS, 'back');
  assert.equal(MOVE_KEYS.KeyD, 'right');
  // Arrows and the usual vertical pair come along for the ride.
  assert.equal(MOVE_KEYS.ArrowUp, 'forward');
  assert.equal(MOVE_KEYS.Space, 'up');
  assert.equal(MOVE_KEYS.KeyQ, 'down');
  for (const code of Object.keys(MOVE_KEYS)) {
    assert.ok(/^(Key[A-Z]|Arrow(Up|Down|Left|Right)|Space)$/.test(code),
      `${code} is not a physical key code`);
  }
});

test('releasing a key stops that direction and leaves the others', () => {
  const { camera, view } = makeCamera();
  view.dispatch('keydown', { code: 'KeyW', target: { tagName: 'BODY' } });
  view.dispatch('keydown', { code: 'KeyD', target: { tagName: 'BODY' } });
  view.dispatch('keyup', { code: 'KeyW' });
  assert.deepEqual([...camera.held], ['right']);
});

test('disposing removes every listener it added', () => {
  // The viewer is created and destroyed on every navigation, so a leak here
  // accumulates cameras that keep flying invisible scenes.
  const { camera, view, document, canvas } = makeCamera();
  assert.ok(view.count('keydown') > 0);
  camera.dispose();
  assert.equal(view.count('keydown'), 0);
  assert.equal(view.count('keyup'), 0);
  assert.equal(view.count('blur'), 0);
  assert.equal(document.count('visibilitychange'), 0);
  assert.equal(canvas.count('pointerdown'), 0);
});

test('the camera holds still when nothing is held', () => {
  const { camera } = makeCamera();
  const before = [...camera.target];
  assert.equal(camera.applyMovement(0.5), false);
  assert.deepEqual(camera.target, before);
});
