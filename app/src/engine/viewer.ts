/**
 * The viewport: a PlayCanvas app that renders the splat and an orbit camera.
 *
 * Two deliberate choices here.
 *
 * `reorder: false` on the asset. PlayCanvas reorders splats on load for memory
 * locality, which is a good default and fatal for us: the engine process talks
 * about splats by index, so both sides must agree on what index 12345 means.
 * We keep file order and accept slightly worse locality.
 *
 * The stage is neutral black, not the warm panel colour. Judging a repair means
 * judging colour, and a tinted background biases that.
 */
import * as pc from 'playcanvas';
import { pickDepth, robustBounds, type NavData, type Bounds } from './navpick';

export interface SplatInfo {
  numSplats: number;
  shBands: number;
  centre: pc.Vec3;
  radius: number;
}

/**
 * Which mouse buttons drive the camera.
 *
 * Both share the parts that never conflict with a tool: the wheel zooms to the
 * cursor, Alt + a button works the way Maya and Houdini do (left turns,
 * middle pans, right zooms), and the right button held with W A S D flies.
 * They differ in the free buttons:
 *
 * - blender: middle turns, Shift + middle pans, Ctrl + middle zooms. The right
 *   button turns too, which is what this app always did.
 * - supersplat: right pans, middle zooms. Left turns in "Orbit" in both.
 */
export type NavPreset = 'blender' | 'supersplat';
type Gesture = 'orbit' | 'pan' | 'zoom' | 'look';

/** A camera pose in the up frame. Angles in degrees. */
export interface Pose {
  yaw: number; pitch: number; distance: number;
  target: [number, number, number];
  /** Vertical field of view. Narrow = the orthographic look. */
  fov: number;
}

/** Ordinary lens, and the telephoto that stands in for orthographic. */
export const FOV_PERSP = 45;
/*
 * Not a real orthographic projection, on purpose. Every brush, the lasso and
 * the box work out depth and size from a perspective camera (selection.ts,
 * gpuselect.ts); a true orthographic matrix gives every splat w = 1 and the
 * surface test would see one flat sheet. A 3-degree lens from far away is
 * indistinguishable on screen -- parallels stay within a fraction of a pixel
 * across the scene -- and every tool keeps working unchanged.
 */
export const FOV_ORTHO = 3;

/**
 * Which file axis points up on screen.
 *
 * 3DGS inherits COLMAP's camera frame -- x right, y *down*, z forward -- and
 * every trainer we have seen (Postshot, Brush, Inria, spz from Niantic's
 * converter) writes scenes in it. Viewing those with +Y up shows every scan on
 * its head, so -Y is the default. The camera turns, never the data: the
 * engine, the selection and the gizmo all keep working in file coordinates.
 */
export type UpAxis = '-y' | '+y' | '+z' | '-z';
export const UP_AXES: UpAxis[] = ['-y', '+y', '+z', '-z'];
export const DEFAULT_UP: UpAxis = '-y';
/** The up axis as a direction in file coordinates, for the engine. */
export const UP_VECTOR: Record<UpAxis, [number, number, number]> = {
  '-y': [0, -1, 0], '+y': [0, 1, 0], '+z': [0, 0, 1], '-z': [0, 0, -1]
};

/** Rotation taking the camera rig's +Y onto the chosen file axis. */
export function upQuat(up: UpAxis): pc.Quat {
  const q = new pc.Quat();
  if (up === '-y') q.setFromEulerAngles(180, 0, 0);       // y -> -y, z -> -z
  else if (up === '+z') q.setFromEulerAngles(90, 0, 0);   // y -> +z
  else if (up === '-z') q.setFromEulerAngles(-90, 0, 0);  // y -> -z
  return q;
}

/**
 * Per-splat edit state, carried in an instance stream and applied in the shader.
 *
 * Deleting used to mean rebuilding the whole resource and re-uploading 130 MB.
 * A splat's state is one integer, so instead it lives in its own texture that
 * the work-buffer pass reads: deleting collapses the splat to zero size,
 * selecting tints it. Editing becomes a texture write, and the selection
 * highlight becomes real geometry that occludes properly instead of dots
 * painted over the top.
 */
/*
 * Deleted and selected live in *separate* streams on purpose. A GPU pass cannot
 * read and write the same stream, and the selection is computed on the GPU
 * while deletion is decided on the CPU -- one combined state would mean the
 * selection pass wiping out every deletion each time it ran.
 */
const MODIFY_WGSL = `
// Live transform of the selection. The committed version runs in the engine
// and also turns the SH bands; this preview does not, so view-dependent
// shading lags behind during a drag and snaps right when it is released.
uniform uXfQuat: vec4f;      // (x, y, z, w)
uniform uXfTrans: vec3f;
uniform uXfPivot: vec3f;
uniform uXfScale: f32;

fn xfRot(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
fn xfMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                 a.w * b.w - dot(a.xyz, b.xyz));
}
fn xfActive() -> bool {
    return uniform.uXfScale != 1.0 || uniform.uXfQuat.w != 1.0
        || dot(uniform.uXfTrans, uniform.uXfTrans) > 0.0;
}

fn modifySplatCenter(center: ptr<function, vec3f>) {
    if (loadEditSel().r != 0u && xfActive()) {
        let rel = (*center - uniform.uXfPivot) * uniform.uXfScale;
        *center = uniform.uXfPivot + xfRot(uniform.uXfQuat, rel) + uniform.uXfTrans;
    }
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f,
                            rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
    if (loadEditDel().r != 0u) {
        *scale = vec3f(0.0);
        return;
    }
    if (loadEditSel().r != 0u && xfActive()) {
        *rotation = xfMul(uniform.uXfQuat, *rotation);
        *scale = *scale * uniform.uXfScale;
    }
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    if (loadEditDel().r != 0u) {
        *color = vec4f((*color).rgb, 0.0);
    } else if (loadEditSel().r != 0u) {
        *color = vec4f(mix((*color).rgb, vec3f(0.898, 0.635, 0.384), 0.6), (*color).a);
    }
}
`;

const MODIFY_GLSL = `
uniform vec4 uXfQuat;
uniform vec3 uXfTrans;
uniform vec3 uXfPivot;
uniform float uXfScale;

vec3 xfRot(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
vec4 xfMul(vec4 a, vec4 b) {
    return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                a.w * b.w - dot(a.xyz, b.xyz));
}
bool xfActive() {
    return uXfScale != 1.0 || uXfQuat.w != 1.0 || dot(uXfTrans, uXfTrans) > 0.0;
}

void modifySplatCenter(inout vec3 center) {
    if (loadEditSel().r != 0u && xfActive()) {
        vec3 rel = (center - uXfPivot) * uXfScale;
        center = uXfPivot + xfRot(uXfQuat, rel) + uXfTrans;
    }
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter,
                              inout vec4 rotation, inout vec3 scale) {
    if (loadEditDel().r != 0u) {
        scale = vec3(0.0);
        return;
    }
    if (loadEditSel().r != 0u && xfActive()) {
        rotation = xfMul(uXfQuat, rotation);
        scale = scale * uXfScale;
    }
}
void modifySplatColor(vec3 center, inout vec4 color) {
    if (loadEditDel().r != 0u) {
        color.a = 0.0;
    } else if (loadEditSel().r != 0u) {
        color.rgb = mix(color.rgb, vec3(0.898, 0.635, 0.384), 0.6);
    }
}
`;

export class Viewer {
  app!: pc.AppBase;
  device!: pc.GraphicsDevice;
  private cameraMoved = false;
  camera!: pc.Entity;
  splat: pc.Entity | null = null;
  splatAsset: pc.Asset | null = null;
  editState = false;
  /** Splats a repair generated. Separate entity so the loaded file is never rebuilt. */
  generated: pc.Entity | null = null;
  /** Its asset, kept so the one it replaces can be unloaded (see `dropAsset`). */
  private generatedAsset: pc.Asset | null = null;
  /** Counts loads, so one that finishes after a newer one began is dropped. */
  private loadSerial = 0;
  generatedEditState = false;
  info: SplatInfo | null = null;
  gsplatData: any = null;
  /** Splat centres in file order, for picking. */
  centres: Float32Array | null = null;

  private target = new pc.Vec3();
  private up: UpAxis = DEFAULT_UP;
  private upRot = upQuat(DEFAULT_UP);
  private distance = 10;
  private yaw = 30;
  private pitch = -20;
  private fov = FOV_PERSP;
  /** Orthographic because a view-cube face or a numpad key asked for it. */
  private autoOrtho = false;
  private dragging: Gesture | null = null;
  private drag = {
    button: 0, x0: 0, y0: 0, moved: false, pointerId: 0,
    pivot: null as pc.Vec3 | null, depth: 1, point: new pc.Vec3()
  };
  private last = { x: 0, y: 0 };
  private onFrame: (() => void) | null = null;
  private frameListeners: Array<() => void> = [];
  private anim: { from: Pose; to: Pose; t0: number; dur: number } | null = null;
  private keys = new Set<string>();
  /** Robust extent of the loaded file, for home and fly speed. */
  private homeBounds: Bounds | null = null;
  private lastPose = '';

  /** What picking and framing look at; set by the app once a file is open. */
  navData: NavData | null = null;
  /** The active tool leaves the left button free ("Orbit"). */
  leftNavigates = false;
  preset: NavPreset = 'supersplat';
  /** Fly speed multiplier; the wheel changes it while flying. */
  flySpeed = 1;
  /** Short messages for the user (fly speed, view names). */
  onNotice: ((msg: string) => void) | null = null;

  async init(canvas: HTMLCanvasElement): Promise<string> {
    this.device = await pc.createGraphicsDevice(canvas, {
      deviceTypes: [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2],
      antialias: false,
      depth: true
    });
    // A cap, not the ratio of the display the window happened to start on:
    // `resizeCanvas` takes the smaller of this and the display's own ratio
    // each time. Fixed at start, a window moved to a HiDPI display stayed soft.
    this.device.maxPixelRatio = 2;

    const options = new pc.AppOptions();
    options.graphicsDevice = this.device;
    options.componentSystems = [
      pc.CameraComponentSystem,
      pc.RenderComponentSystem,
      pc.GSplatComponentSystem
    ];
    options.resourceHandlers = [pc.GSplatHandler, pc.TextureHandler];

    const app = new pc.AppBase(canvas);
    app.init(options);
    this.app = app;
    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    this.setSmallSplatCull(0, 0);     // no splat is too small to draw (see there)

    this.camera = new pc.Entity('camera');
    this.camera.addComponent('camera', {
      clearColor: new pc.Color(0.02, 0.02, 0.02, 1),   // neutral stage
      fov: 45,
      nearClip: 0.01,
      farClip: 5000,
      toneMapping: pc.TONEMAP_NONE
    });
    app.root.addChild(this.camera);
    app.start();

    this.attachControls(canvas);
    this.resize(canvas);
    // The display's ratio changes without the viewport changing size when the
    // window is dragged to another screen. A media query for the ratio as it
    // is now fires once when it stops being true; then watch the new one.
    const watchRatio = () => {
      matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener('change', () => {
        this.resize(canvas);
        watchRatio();
      }, { once: true });
    };
    watchRatio();
    app.on('update', (dt: number) => {
      this.step(Math.min(dt, 0.1));
      this.applyCamera();
      this.onFrame?.();
      for (const f of this.frameListeners) f();
    });
    return this.device.isWebGPU ? 'WebGPU' : 'WebGL 2';
  }

  /**
   * How small a splat may get on screen before the renderer leaves it out.
   *
   * PlayCanvas has two such limits, both on by default: a splat under
   * `minPixelSize` pixels across is dropped (2), and with the WebGPU renderer
   * also one whose opacity times projected area stays under `minContribution`
   * (3). A fill or a stamp is made of splats far smaller than the scan's.
   * Zoomed out they fell under both limits long before the scan around them
   * did: the patch thinned out and what lies behind it showed through. Both
   * are off here (0), so a splat is drawn however small it gets. (On WebGPU
   * the second is the one that bites: with the pixel limit alone at 0 the
   * fill over the shoe's stand was as see-through from eight radii away as
   * with both on.)
   *
   * The limits belong to the scene, not to an entity (`scene.gsplat`, the
   * parameters the unified renderer draws every splat with): the scan and
   * the generated splats share them, and a new file or a generated set that
   * is built again needs nothing set a second time.
   */
  setSmallSplatCull(minPixelSize: number, minContribution: number) {
    const g = this.app.scene.gsplat;
    g.minPixelSize = minPixelSize;
    g.minContribution = minContribution;
  }

  /** The two limits as they stand (tests). */
  get smallSplatCull(): { minPixelSize: number; minContribution: number } {
    const g = this.app.scene.gsplat;
    return { minPixelSize: g.minPixelSize, minContribution: g.minContribution };
  }

  onRender(fn: () => void) { this.onFrame = fn; }
  /** Called every frame after the camera is placed (the view cube draws here). */
  addFrameListener(fn: () => void) { this.frameListeners.push(fn); }

  resize(canvas: HTMLCanvasElement) {
    // Measure the *container*, not the canvas.
    //
    // `app.resizeCanvas` writes inline style.width/height on the canvas every
    // time it is called. From the first call onwards the canvas is a
    // fixed-pixel block, so asking the canvas how big it is just reads that
    // frozen number back -- and going fullscreen grew the container while the
    // canvas stayed put, leaving the render in the top-left corner with black
    // around it. Measuring the parent and resizing the device directly
    // touches only the backing store and leaves the CSS to the stylesheet.
    const box = (canvas.parentElement ?? canvas).getBoundingClientRect();
    const w = Math.max(1, box.width | 0), h = Math.max(1, box.height | 0);
    canvas.style.removeProperty('width');
    canvas.style.removeProperty('height');
    this.device?.resizeCanvas(w, h);
    this.cameraMoved = true;
  }

  /**
   * Take an asset out of the registry and free what it holds.
   *
   * Destroying the entity alone left the asset registered and its resource
   * alive: the textures of every file opened and of every generated set an
   * edit replaced stayed on the GPU and in RAM for the whole session. The
   * entity goes first; a resource the renderer still draws from this frame
   * is counted by PlayCanvas and destroyed when it lets go.
   */
  private dropAsset(asset: pc.Asset | null) {
    if (!asset) return;
    this.app.assets.remove(asset);
    asset.unload();
  }

  /** Load a splat, replacing whatever is loaded. */
  async load(url: string, filename: string): Promise<SplatInfo> {
    const serial = ++this.loadSerial;
    this.clear();
    const asset = new pc.Asset(filename, 'gsplat', { url, filename },
      { reorder: false });           // index order must match the engine process
    this.app.assets.add(asset);

    try {
      await new Promise<void>((resolve, reject) => {
        asset.once('load', () => resolve());
        asset.once('error', (e: string) => reject(new Error(e)));
        this.app.assets.load(asset);
      });
    } catch (e) {
      this.dropAsset(asset);
      throw e;
    }
    if (serial !== this.loadSerial) {
      // Another load began while this file was being read. Its splats must
      // not appear beside the newer file's (they used to, as an entity
      // nothing referred to any more).
      this.dropAsset(asset);
      throw new Error('Loading canceled: another file is being loaded.');
    }

    const entity = new pc.Entity(filename);
    entity.addComponent('gsplat', { asset });
    this.app.root.addChild(entity);
    this.splat = entity;
    this.splatAsset = asset;
    this.editState = this.enableEditState(entity, asset);
    this.clearTransformPreview();

    const data = (asset.resource as any)?.gsplatData;
    this.gsplatData = data ?? null;
    const n = data?.numSplats ?? 0;
    this.centres = data?.getCenters?.() ?? null;

    const aabb = new pc.BoundingBox();
    if (data?.calcAabb) data.calcAabb(aabb);
    const centre = aabb.center.clone();
    const radius = Math.max(aabb.halfExtents.length(), 1e-3);

    this.info = { numSplats: n, shBands: data?.shBands ?? 0, centre, radius };
    this.homeBounds = this.centres ? robustBounds(this.centres, null) : null;
    this.frameScene();
    return this.info;
  }

  /**
   * Replace the generated splats. Pass null to remove them.
   * The loaded file is untouched by this -- only the much smaller set a repair
   * produced is re-uploaded, and only when it actually changed.
   */
  setGenerated(data: any | null) {
    const old = this.generated, oldAsset = this.generatedAsset;
    this.generated = null;
    this.generatedAsset = null;
    if (data) {
      const resource = new (pc as any).GSplatResource(this.device, data);
      const asset = new pc.Asset('generated', 'gsplat', null as any);
      (asset as any).resource = resource;
      (asset as any).loaded = true;
      this.app.assets.add(asset);
      const entity = new pc.Entity('generated');
      entity.addComponent('gsplat', { asset });
      this.app.root.addChild(entity);
      this.generated = entity;
      this.generatedAsset = asset;
      this.generatedEditState = this.enableEditState(entity, asset);
      this.clearTransformPreview();
    }
    if (old) old.destroy();
    this.dropAsset(oldAsset);
  }

  clear() {
    if (this.generated) {
      this.generated.destroy();
      this.generated = null;
    }
    this.dropAsset(this.generatedAsset);
    this.generatedAsset = null;
    if (this.splat) {
      this.splat.destroy();
      this.splat = null;
    }
    this.dropAsset(this.splatAsset);
    this.splatAsset = null;
    this.info = null;
    this.gsplatData = null;
    this.centres = null;
  }

  /** Row-major view-projection matrix, for bulk projection of splat centres. */
  viewProjection(): Float32Array {
    // Built from the camera as it stands now, not from the matrices the
    // camera component caches: those are refreshed when a frame renders, and
    // a window that is not being drawn (behind others, during a long
    // command) kept handing out the previous view. A stroke aimed right
    // after a camera move then projected through the old camera.
    const cam = this.camera.camera as any;
    const r = (this.app.graphicsDevice.canvas as HTMLCanvasElement).getBoundingClientRect();
    const aspect = r.height > 0 ? r.width / r.height : 1;
    const proj = new pc.Mat4().setPerspective(cam.fov, aspect, cam.nearClip, cam.farClip);
    const view = this.camera.getWorldTransform().clone().invert();
    return new pc.Mat4().mul2(proj, view).data as Float32Array;
  }

  /** The standard view, at once: what a file opens with. */
  frameScene() {
    if (!this.info) return;
    this.anim = null;
    this.autoOrtho = false;
    this.setPose(this.homePose(false));
  }

  /** The axis that points up on screen. Orbit, pan and the gizmo follow it. */
  get upAxis(): UpAxis { return this.up; }

  setUpAxis(up: UpAxis) {
    this.up = up;
    this.upRot = upQuat(up);
    this.cameraMoved = true;
    this.frameScene();
  }

  /** The up frame as a rotation, for handles that should stand upright too. */
  upRotation(): pc.Quat { return this.upRot.clone(); }

  /**
   * Give an entity an `editState` stream and the shader that reads it.
   * Returns false if the engine would not take it, so callers can fall back.
   */
  enableEditState(entity: pc.Entity, asset: pc.Asset): boolean {
    try {
      const resource = (asset as any).resource;
      const P = pc as any;
      resource.format.addExtraStreams([
        { name: 'editDel', format: P.PIXELFORMAT_R32U, storage: P.GSPLAT_STREAM_INSTANCE },
        // What is already selected, and what the live stroke makes of it. Two
        // streams because the GPU pass has to read the first to honour "add"
        // and "subtract" while writing the second.
        { name: 'editBase', format: P.PIXELFORMAT_R32U, storage: P.GSPLAT_STREAM_INSTANCE },
        { name: 'editSel', format: P.PIXELFORMAT_R32U, storage: P.GSPLAT_STREAM_INSTANCE },
        // Covariance as float bits, for the size-aware brush (gpuselect.ts).
        // Unsigned formats because 32-bit float textures other than RGBA32F
        // are not bindable without the float32-filterable feature on WebGPU.
        { name: 'editCovA', format: P.PIXELFORMAT_RGBA32U, storage: P.GSPLAT_STREAM_INSTANCE },
        { name: 'editCovB', format: P.PIXELFORMAT_RG32U, storage: P.GSPLAT_STREAM_INSTANCE }
      ]);
      (entity.gsplat as any).setWorkBufferModifier({
        wgsl: MODIFY_WGSL, glsl: MODIFY_GLSL
      });
      return true;
    } catch (e) {
      console.warn('editState stream unavailable:', e);
      return false;
    }
  }

  /**
   * Preview a transform on the selection, without touching any splat data.
   *
   * Dragging a gizmo this way costs four uniforms per frame instead of
   * rewriting positions; the real transform is applied by the engine once the
   * handle is let go.
   */
  setTransformPreview(quat: [number, number, number, number],
                      trans: [number, number, number],
                      pivot: [number, number, number], scale: number) {
    for (const e of [this.splat, this.generated]) {
      const g = (e as any)?.gsplat;
      if (!g) continue;
      g.setParameter('uXfQuat', quat);
      g.setParameter('uXfTrans', trans);
      g.setParameter('uXfPivot', pivot);
      g.setParameter('uXfScale', scale);
    }
  }

  clearTransformPreview() {
    this.setTransformPreview([0, 0, 0, 1], [0, 0, 0], [0, 0, 0], 1);
  }

  /** Upload one per-splat flag stream. `states[i]` belongs to splat `i`. */
  writeStream(entity: pc.Entity,
              stream: 'editDel' | 'editBase' | 'editSel' | 'editCovA' | 'editCovB',
              states: Uint32Array<ArrayBufferLike>): boolean {
    const tex = (entity.gsplat as any)?.getInstanceTexture?.(stream);
    if (!tex) return false;
    const data = tex.lock() as Uint32Array;
    // The texture is square-ish and larger than the splat count; the tail is
    // padding that no splat reads, so only the first n entries matter.
    data.fill(0);
    data.set(states.subarray(0, Math.min(states.length, data.length)));
    tex.unlock();
    // Evaluated splats are cached in a work buffer that is only refreshed when
    // the camera turns far enough. Without this the texture is correct and the
    // picture does not change, which looks exactly like a broken shader.
    (entity.gsplat as any)?._placement?.markDirty?.();
    return true;
  }

  /** Diagnostic: tint every splat, to prove the modifier hook is live at all. */
  forceTintAll(entity: pc.Entity) {
    (entity.gsplat as any).setWorkBufferModifier({
      wgsl: `
fn modifySplatCenter(center: ptr<function, vec3f>) {}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f,
                            rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    *color = vec4f(1.0, 0.0, 0.0, (*color).a);
}`,
      glsl: `
void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 o, vec3 m, inout vec4 r, inout vec3 s) {}
void modifySplatColor(vec3 center, inout vec4 color) { color.rgb = vec3(1.0, 0.0, 0.0); }`
    });
    (entity.gsplat as any)?._placement?.markDirty?.();
  }

  /**
   * Set the orbit directly, for tests: about the file's bounding-box centre,
   * `distance` in bounding-box radii, perspective lens. The target is reset
   * too, so a test framing does not depend on where anyone navigated before.
   */
  setView(yaw?: number, pitch?: number, distance?: number) {
    this.anim = null;
    this.autoOrtho = false;
    this.fov = FOV_PERSP;
    if (this.info) this.target.copy(this.info.centre);
    if (yaw !== undefined) this.yaw = yaw;
    if (pitch !== undefined) this.pitch = pitch;
    if (distance !== undefined && this.info) this.distance = this.info.radius * distance;
    this.applyCamera();
  }

  /* ------------------------------------------------------------ poses */

  pose(): Pose {
    return { yaw: this.yaw, pitch: this.pitch, distance: this.distance,
             target: [this.target.x, this.target.y, this.target.z], fov: this.fov };
  }

  private setPose(p: Pose) {
    this.yaw = p.yaw; this.pitch = p.pitch; this.distance = p.distance;
    this.target.set(p.target[0], p.target[1], p.target[2]);
    this.fov = p.fov;
    this.applyCamera();
  }

  /**
   * Glide to a pose. A third of a second, easing out: long enough to see
   * where the view went, short enough not to wait for it. Yaw takes the short
   * way round. Reduced motion jumps.
   */
  animateTo(to: Pose, dur = 320) {
    const from = this.pose();
    let dy = ((to.yaw - from.yaw) % 360 + 540) % 360 - 180;
    let dp = ((to.pitch - from.pitch) % 360 + 540) % 360 - 180;
    const goal = { ...to, yaw: from.yaw + dy, pitch: from.pitch + dp };
    const still = typeof matchMedia === 'function'
      && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (still || dur <= 0) { this.anim = null; this.setPose(goal); return; }
    this.anim = { from, to: goal, t0: performance.now(), dur };
  }

  /** Is a view transition running? (For tests: wait until false.) */
  get isAnimating() { return this.anim !== null; }

  /** Distance at which a sphere of `radius` fills the view, with a margin. */
  private fitDistance(radius: number, fov = this.fov): number {
    const r = (this.app?.graphicsDevice?.canvas as HTMLCanvasElement | undefined)
      ?.getBoundingClientRect();
    const aspect = r && r.height > 0 ? r.width / r.height : 1.5;
    const tv = Math.tan(fov * Math.PI / 360);
    const th = tv * aspect;
    const half = Math.atan(Math.min(tv, th));
    return radius / Math.sin(half) * 1.08;
  }

  /** The standard view: a little from above, the bulk of the scan filling it. */
  homePose(current = true): Pose {
    // While a file loads, the app's data still describes the previous one:
    // the file's own extent then, measured from its centres at load.
    const b = (current ? this.currentBounds() : null) ?? this.homeBounds;
    const c = b ? b.centre : this.info ? [this.info.centre.x, this.info.centre.y, this.info.centre.z] as [number, number, number] : [0, 0, 0] as [number, number, number];
    const radius = b ? b.radius : this.info?.radius ?? 1;
    // A little from above, the way a product shot is framed. It used to be
    // from below (-22), which on a y-down scan happened to look down on it.
    return { yaw: 30, pitch: 22, distance: this.fitDistance(radius, FOV_PERSP),
             target: c, fov: FOV_PERSP };
  }

  /** Where the visible splats are now, robustly; null before a file is open. */
  private currentBounds(): Bounds | null {
    if (!this.navData) return null;
    return robustBounds(this.navData.centres(), this.navData.visible());
  }

  /** Back to the standard view (Home, the house button). */
  home(animate = true) {
    if (!this.info) return;
    this.autoOrtho = false;
    const p = this.homePose();
    if (animate) this.animateTo(p); else { this.anim = null; this.setPose(p); }
  }

  /** Fit a sphere into the view from the current angle. */
  frameSphere(centre: [number, number, number], radius: number, animate = true) {
    const p = { ...this.pose(), target: centre, distance: this.fitDistance(radius) };
    if (animate) this.animateTo(p); else { this.anim = null; this.setPose(p); }
  }

  /** Everything visible, from the current angle ("." in Blender). */
  frameAll(animate = true) {
    const b = this.currentBounds() ?? this.homeBounds;
    if (b) this.frameSphere(b.centre, b.radius, animate);
  }

  /** Scene size for speeds: the robust radius of the file. */
  get sceneRadius(): number { return this.homeBounds?.radius ?? this.info?.radius ?? 1; }

  get isOrtho() { return this.fov < 10; }

  /**
   * Switch between the ordinary lens and the orthographic look, keeping the
   * plane through the target the same size on screen.
   */
  setOrtho(on: boolean, animate = true) {
    const fov = on ? FOV_ORTHO : FOV_PERSP;
    if (Math.abs(fov - this.fov) < 1e-6 && !this.anim) return;
    const base = this.anim ? this.anim.to : this.pose();
    const k = Math.tan(base.fov * Math.PI / 360) / Math.tan(fov * Math.PI / 360);
    const p = { ...base, fov, distance: base.distance * k };
    if (!on) this.autoOrtho = false;
    if (animate) this.animateTo(p, 260); else { this.anim = null; this.setPose(p); }
  }

  /** Leave an orthographic view that a snap set, the way Blender does when you turn. */
  private leaveAutoOrtho() {
    if (!this.autoOrtho || !this.isOrtho) { this.autoOrtho = false; return; }
    this.autoOrtho = false;
    this.anim = null;
    const k = Math.tan(this.fov * Math.PI / 360) / Math.tan(FOV_PERSP * Math.PI / 360);
    // Keep the eye where the target plane is the same size: a dolly zoom in one frame.
    this.distance *= k;
    this.fov = FOV_PERSP;
    this.applyCamera();
  }

  /**
   * Look from a direction (in the up frame: +x right, +y up, +z front),
   * pointing at the current target. Faces go orthographic, edges and
   * corners stay in perspective. Asking for the view you already have turns
   * to the opposite side, as Blender's view cube does.
   */
  snapTo(dir: [number, number, number], ortho?: boolean) {
    let [x, y, z] = dir;
    const len = Math.hypot(x, y, z) || 1;
    x /= len; y /= len; z /= len;
    const cur = this.viewDirLocal();
    if (cur[0] * x + cur[1] * y + cur[2] * z > 0.9995) { x = -x; y = -y; z = -z; }
    const axisView = [x, y, z].filter(v => Math.abs(v) > 1e-6).length === 1;
    const wantOrtho = ortho ?? axisView;
    const pitch = Math.asin(Math.max(-1, Math.min(1, y))) * 180 / Math.PI;
    // Straight up or down has no yaw of its own; keep the nearest quarter turn
    // of the current one so the picture does not spin.
    const yaw = Math.abs(y) > 0.9999 ? Math.round(this.yaw / 90) * 90
      : Math.atan2(x, z) * 180 / Math.PI;
    const base = this.anim ? this.anim.to : this.pose();
    const fov = wantOrtho ? FOV_ORTHO : FOV_PERSP;
    const k = Math.tan(base.fov * Math.PI / 360) / Math.tan(fov * Math.PI / 360);
    this.autoOrtho = wantOrtho;
    this.animateTo({ ...base, yaw, pitch, fov, distance: base.distance * k });
  }

  /** Direction from the target to the camera, in the up frame. */
  viewDirLocal(): [number, number, number] {
    const q = new pc.Quat().setFromEulerAngles(-this.pitch, this.yaw, 0);
    const b = q.transformVector(new pc.Vec3(0, 0, 1));
    return [b.x, b.y, b.z];
  }

  /** The camera's turn in the up frame (for the view cube). */
  localRotation(): pc.Quat {
    return new pc.Quat().setFromEulerAngles(-this.pitch, this.yaw, 0);
  }

  /** Turn about the target, as dragging the view cube does. */
  orbitDrag(dx: number, dy: number) {
    this.anim = null;
    this.leaveAutoOrtho();
    this.turn(dx, dy, null);
  }

  /** Centre the view on a world point without turning or zooming. */
  centreOn(p: [number, number, number], animate = true) {
    const eye = this.camera.getPosition();
    const f = this.camera.forward;
    const d = (p[0] - eye.x) * f.x + (p[1] - eye.y) * f.y + (p[2] - eye.z) * f.z;
    const to = { ...this.pose(), target: p, distance: d > 1e-6 ? d : this.distance };
    if (animate) this.animateTo(to, 260); else { this.anim = null; this.setPose(to); }
  }

  /* --------------------------------------------------------- navigation */

  private orientation(): pc.Quat {
    const pitch = ((this.pitch + 180) % 360 + 360) % 360 - 180;
    return new pc.Quat().mul2(this.upRot, new pc.Quat().setFromEulerAngles(-pitch, this.yaw, 0));
  }

  /**
   * Turn by a mouse delta, about `pivot` (null: the target). The pivot stays
   * where it is on screen: the camera's offset from it turns with the camera.
   */
  private turn(dx: number, dy: number, pivot: pc.Vec3 | null, rate = 0.3) {
    // Past the pole the world is upside down, so a drag to the right
    // would turn the model to the left. Flip with it.
    const flip = Math.cos(this.pitch * Math.PI / 180) < 0 ? -1 : 1;
    const q0 = this.camera.getRotation().clone();
    const eye0 = this.camera.getPosition().clone();
    this.yaw -= dx * rate * flip;
    this.pitch -= dy * rate;
    if (pivot) {
      const q1 = this.orientation();
      const dq = new pc.Quat().mul2(q1, q0.invert());
      const off = new pc.Vec3().sub2(eye0, pivot);
      dq.transformVector(off, off);
      const eye1 = off.add(pivot);
      const fwd1 = q1.transformVector(new pc.Vec3(0, 0, -1));
      this.target.copy(eye1).add(fwd1.mulScalar(this.distance));
    }
    this.applyCamera();
  }

  /** Move the camera so `p` stays put on screen and the eye gets `f` times as far from it. */
  private zoomAt(p: pc.Vec3, f: number) {
    const eye = this.camera.getPosition().clone();
    const fwd = this.camera.forward.clone();
    const depth = new pc.Vec3().sub2(p, eye).dot(fwd);
    if (!(depth > 1e-9)) return;
    const minD = this.sceneRadius * 1e-4;
    const k = Math.max(f, minD / depth);
    const eye1 = new pc.Vec3().sub2(eye, p).mulScalar(k).add(p);
    this.distance = depth * k;
    this.target.copy(eye1).add(fwd.mulScalar(this.distance));
    this.applyCamera();
  }

  /** Straight ahead by a world distance, through surfaces if need be. */
  private dolly(step: number) {
    const fwd = this.camera.forward.clone();
    this.target.add(fwd.mulScalar(step));
    this.applyCamera();
  }

  /** The world point under a viewport pixel at a camera depth. */
  private rayPoint(x: number, y: number, depth: number): pc.Vec3 {
    const w = this.unproject(x, y, depth);
    return new pc.Vec3(w[0], w[1], w[2]);
  }

  /** Camera depth of the surface under a viewport pixel, or null over empty space. */
  surfaceDepthAt(x: number, y: number): number | null {
    const data: NavData | null = this.navData ?? (this.centres ? {
      centres: () => this.centres!, visible: () => null, alpha: () => null, minAlpha: () => 0
    } : null);
    if (!data) return null;
    const r = (this.app.graphicsDevice.canvas as HTMLCanvasElement).getBoundingClientRect();
    const e = this.camera.getPosition(), rt = this.camera.right, up = this.camera.up,
      fw = this.camera.forward;
    return pickDepth(data, {
      eye: [e.x, e.y, e.z], right: [rt.x, rt.y, rt.z], up: [up.x, up.y, up.z],
      fwd: [fw.x, fw.y, fw.z], tanHalf: Math.tan(this.fov * Math.PI / 360),
      width: r.width, height: r.height
    }, x, y);
  }

  /** One wheel notch at a viewport pixel, as the wheel does. Exposed for tests. */
  wheelAt(x: number, y: number, deltaY: number, mods: { ctrl?: boolean; shift?: boolean } = {}) {
    this.anim = null;
    if (mods.ctrl) {
      // through the surface: a fixed step, which the zoom never takes
      this.dolly(-Math.sign(deltaY) * this.sceneRadius * 0.04);
      return;
    }
    const d = this.surfaceDepthAt(x, y);
    const p = this.rayPoint(x, y, d ?? this.distance);
    const k = mods.shift ? 0.0005 : 0.0018;
    this.zoomAt(p, Math.exp(Math.max(-300, Math.min(300, deltaY)) * k));
  }

  /** Fly: move with W A S D (Q down, E up) while the right button is held. */
  private step(dt: number) {
    if (this.anim) {
      const a = this.anim;
      const t = Math.min(1, (performance.now() - a.t0) / a.dur);
      const e = 1 - Math.pow(1 - t, 3);
      const L = (u: number, v: number) => u + (v - u) * e;
      // distance in log space: a long zoom-out should not crawl at the start
      const dist = Math.exp(L(Math.log(a.from.distance), Math.log(a.to.distance)));
      this.yaw = L(a.from.yaw, a.to.yaw);
      this.pitch = L(a.from.pitch, a.to.pitch);
      this.distance = dist;
      this.fov = L(a.from.fov, a.to.fov);
      this.target.set(L(a.from.target[0], a.to.target[0]), L(a.from.target[1], a.to.target[1]),
                      L(a.from.target[2], a.to.target[2]));
      if (t >= 1) this.anim = null;
    }
    if (this.dragging !== 'look' || !this.keys.size) return;
    const k = this.keys;
    const f = (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0);
    const s = (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0);
    const u = (k.has('KeyE') ? 1 : 0) - (k.has('KeyQ') ? 1 : 0);
    if (!f && !s && !u) return;
    const fast = k.has('ShiftLeft') || k.has('ShiftRight') ? 3 : 1;
    const v = this.sceneRadius * 0.6 * this.flySpeed * fast * dt;
    const upW = this.upRot.transformVector(new pc.Vec3(0, 1, 0));
    const move = this.camera.forward.clone().mulScalar(f * v)
      .add(this.camera.right.clone().mulScalar(s * v))
      .add(upW.mulScalar(u * v));
    this.target.add(move);
  }

  private applyCamera() {
    // Rotation is unrestricted: the turntable goes over the pole and keeps
    // going, like a model in the hand.
    //
    // It used to clamp pitch to +-89.5 -- but only on a local copy, while the
    // accumulator kept counting. Drag past the pole and the picture froze
    // while `pitch` ran off to 250 degrees; dragging back did nothing until
    // all of that had been unwound. That is the "hits a limit and sticks".
    //
    // Clamping the accumulator instead would fix the stickiness but keep the
    // limit, and `lookAt` with a fixed world up is singular at the pole
    // anyway. Setting the Euler angles directly has no singularity at all,
    // and with roll pinned to zero the horizon stays level.
    //
    // Yaw and pitch live in the up frame: yaw turns about the scene's up axis,
    // pitch about the camera's right. Rotating the whole rig by the up frame
    // keeps roll at zero relative to the scene's horizon, whichever file axis
    // that is.
    this.pitch = ((this.pitch + 180) % 360 + 360) % 360 - 180;
    const local = new pc.Quat().setFromEulerAngles(-this.pitch, this.yaw, 0);
    this.camera.setRotation(new pc.Quat().mul2(this.upRot, local));
    const eye = this.camera.forward.clone().mulScalar(-this.distance)
      .add(this.target);
    this.camera.setPosition(eye);
    const cam = this.camera.camera;
    if (cam) {
      if (cam.fov !== this.fov) cam.fov = this.fov;
      // The orthographic look sits far back; the far plane has to follow it.
      const far = Math.max(5000, (this.distance + this.sceneRadius * 4) * 2);
      if (cam.farClip < far) cam.farClip = far;
      const near = Math.min(0.01, Math.max(this.distance * 1e-3, 1e-5));
      if (Math.abs(cam.nearClip - near) > 1e-9) cam.nearClip = near;
    }
    // Everything that depends on the camera (the brush's projection, the
    // stamp marker, the view cube) keys off this, whatever moved it.
    const key = `${eye.x},${eye.y},${eye.z},${this.yaw},${this.pitch},${this.fov}`;
    if (key !== this.lastPose) { this.lastPose = key; this.cameraMoved = true; }
  }

  /** World units per viewport pixel at a given camera-space depth. */
  worldPerPixelAt(depth: number): number {
    return depth * this.worldPerPixelPerUnit();
  }

  /**
   * World units per pixel at one unit of depth -- the whole thing is linear
   * in depth, so this is the only part worth computing.
   *
   * Worth having separately because the selection loops wanted it per splat,
   * and the general form reads `getBoundingClientRect()` every call. Half a
   * million forced layouts per lasso is what turned a 20 ms selection into a
   * 96 ms one.
   */
  worldPerPixelPerUnit(): number {
    const c = this.camera.camera as any;
    const r = (this.app.graphicsDevice.canvas as HTMLCanvasElement)
      .getBoundingClientRect();
    const fov = (c.fov ?? 45) * Math.PI / 180;
    return 2 * Math.tan(fov * 0.5) / Math.max(r.height, 1);
  }

  /** Screen pixel + camera-space depth back to a world point. */
  unproject(x: number, y: number, depth: number): [number, number, number] {
    const c = this.camera.camera as any;
    const r = (this.app.graphicsDevice.canvas as HTMLCanvasElement).getBoundingClientRect();
    const fov = (c.fov ?? 45) * Math.PI / 180;
    const t = Math.tan(fov * 0.5);
    const aspect = r.width / Math.max(r.height, 1);
    const nx = (2 * x / Math.max(r.width, 1) - 1) * t * aspect;
    const ny = (1 - 2 * y / Math.max(r.height, 1)) * t;
    const e = this.camera.getPosition();
    const rt = this.camera.right, up = this.camera.up, fw = this.camera.forward;
    return [
      e.x + (rt.x * nx + up.x * ny + fw.x) * depth,
      e.y + (rt.y * nx + up.y * ny + fw.y) * depth,
      e.z + (rt.z * nx + up.z * ny + fw.z) * depth
    ];
  }

  /** Unit view direction, for flattening the brush towards the camera. */
  viewDirection(): [number, number, number] {
    const f = this.camera.forward;
    return [f.x, f.y, f.z];
  }

  /** True while the user is driving the camera, so tools can stand aside. */
  get isNavigating() { return this.dragging !== null; }
  /** Flying with the right button and W A S D. */
  get isFlying() { return this.dragging === 'look'; }

  /** Has the camera moved since this was last asked? Clears on read. */
  consumeCameraMoved(): boolean {
    const m = this.cameraMoved;
    this.cameraMoved = false;
    return m;
  }

  /**
   * Which gesture a button press starts, or null when the tool owns it.
   *
   * The left button belongs to the active tool except in "Orbit" and with
   * Alt, so a brush never has to share it with the camera.
   */
  private gestureFor(e: PointerEvent): Gesture | null {
    const b = e.button, sh = e.shiftKey, ct = e.ctrlKey || e.metaKey;
    const mod = (g: Gesture): Gesture => sh ? 'pan' : ct ? 'zoom' : g;
    if (e.altKey) {
      // Maya and Houdini: left turns, middle pans, right zooms
      if (b === 0) return mod('orbit');
      if (b === 1) return 'pan';
      if (b === 2) return 'zoom';
    }
    if (b === 0) return this.leftNavigates ? mod('orbit') : null;
    if (this.preset === 'supersplat') {
      if (b === 2) return 'pan';
      if (b === 1) return 'zoom';
      return null;
    }
    if (b === 1 || b === 2) return mod('orbit');
    return null;
  }

  private static FLY_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);

  private attachControls(canvas: HTMLCanvasElement) {
    const local = (e: { clientX: number; clientY: number }) => {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const flyKeyHeld = () => [...this.keys].some(k => Viewer.FLY_KEYS.has(k));

    canvas.addEventListener('pointerdown', (e) => {
      if (this.dragging) return;
      let g = this.gestureFor(e);
      if (!g) return;
      // the right button with a fly key already down flies from the start
      if (e.button === 2 && flyKeyHeld()) g = 'look';
      this.anim = null;
      const p = local(e);
      this.dragging = g;
      Object.assign(this.drag, { button: e.button, x0: p.x, y0: p.y, moved: false,
                                 pointerId: e.pointerId, pivot: null });
      this.last = { x: e.clientX, y: e.clientY };
      this.beginGesture(g, p.x, p.y);
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.last.x;
      const dy = e.clientY - this.last.y;
      this.last = { x: e.clientX, y: e.clientY };
      const p = local(e);
      if (!this.drag.moved && Math.hypot(p.x - this.drag.x0, p.y - this.drag.y0) < 3) return;
      if (!this.drag.moved) {
        this.drag.moved = true;
        if (this.dragging === 'orbit') this.leaveAutoOrtho();
      }
      if (this.dragging === 'orbit') {
        this.turn(dx, this.preset === 'supersplat' ? -dy : dy, this.drag.pivot);
      } else if (this.dragging === 'look') {
        // fly: look around from where the camera is
        this.turn(dx, -dy, this.camera.getPosition().clone(), 0.18);
      } else if (this.dragging === 'pan') {
        // the grabbed point stays under the hand
        const s = this.drag.depth * this.worldPerPixelPerUnit();
        const right = this.camera.right.clone().mulScalar(-dx * s);
        const up = this.camera.up.clone().mulScalar(dy * s);
        this.target.add(right).add(up);
        this.applyCamera();
      } else {
        this.zoomAt(this.drag.point, Math.exp((dy - dx) * 0.005));
      }
    });

    const end = (e: PointerEvent) => {
      if (!this.dragging) return;
      const g = this.dragging, moved = this.drag.moved, button = this.drag.button;
      this.dragging = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
      // a middle click that did not drag centres on the point under it
      if (!moved && button === 1 && g !== 'look') this.centreAtPixel(this.drag.x0, this.drag.y0);
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    // no autoscroll compass on the middle button
    canvas.addEventListener('mousedown', (e) => { if (e.button === 1) e.preventDefault(); });
    canvas.addEventListener('auxclick', (e) => e.preventDefault());

    canvas.addEventListener('dblclick', (e) => {
      if (!this.leftNavigates) return;
      const p = local(e);
      this.centreAtPixel(p.x, p.y);
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const p = local(e);
      // A notch is 100 px on Windows; lines and pages are scaled to match.
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 300 : e.deltaY;
      if (this.dragging === 'look' || (this.dragging && this.drag.button === 2 && flyKeyHeld())) {
        this.flySpeed = Math.max(0.05, Math.min(40, this.flySpeed * Math.exp(-Math.sign(dy) * 0.2)));
        this.onNotice?.(`Fly speed ${this.flySpeed.toLocaleString('en-US', { maximumFractionDigits: 2 })} ×`);
        return;
      }
      if (this.dragging) return;
      this.wheelAt(p.x, p.y, dy, { ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey });
    }, { passive: false });

    // Fly keys. Capture phase on the window, so they reach this before the
    // tool shortcuts (S is the stamp) -- and only taken while the right
    // button is down, so the letters stay the tools' otherwise.
    const typing = (t: EventTarget | null) => t instanceof HTMLElement
      && (t.tagName === 'INPUT' && (t as HTMLInputElement).type !== 'range'
          || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    window.addEventListener('keydown', (e) => {
      if (typing(e.target)) return;
      this.keys.add(e.code);
      if (!Viewer.FLY_KEYS.has(e.code)) return;
      if (this.dragging && this.drag.button === 2) {
        if (this.dragging !== 'look') { this.dragging = 'look'; this.drag.moved = true; this.leaveAutoOrtho(); }
        e.preventDefault();
        e.stopPropagation();
      }
    }, true);
    window.addEventListener('keyup', (e) => { this.keys.delete(e.code); }, true);
    window.addEventListener('blur', () => this.keys.clear());
  }

  /** Set up what a gesture works around: the surface under the cursor. */
  private beginGesture(g: Gesture, x: number, y: number) {
    if (g === 'look') return;
    if (g === 'orbit' && this.preset === 'supersplat') {
      // Keep the focus chosen by framing, panning or a double-click.
      this.drag.pivot = null;
      return;
    }
    const d = this.surfaceDepthAt(x, y);
    if (g === 'orbit') {
      if (d !== null) {
        // Turn about the surface point under the cursor. The focus moves
        // there without moving the eye, so nothing jumps.
        const eye = this.camera.getPosition().clone();
        this.distance = d;
        this.target.copy(eye).add(this.camera.forward.clone().mulScalar(d));
        this.drag.pivot = this.rayPoint(x, y, d);
        this.applyCamera();
      } else {
        this.drag.pivot = null;       // empty space: about the last focus
      }
    } else if (g === 'pan') {
      this.drag.depth = d ?? this.distance;
    } else {
      this.drag.point = this.rayPoint(x, y, d ?? this.distance);
    }
  }

  /** Centre the view on the surface under a pixel; nothing there, nothing happens. */
  centreAtPixel(x: number, y: number): boolean {
    const d = this.surfaceDepthAt(x, y);
    if (d === null) return false;
    const p = this.rayPoint(x, y, d);
    this.centreOn([p.x, p.y, p.z]);
    return true;
  }

  /** Project a world point to viewport CSS pixels. */
  worldToViewport(x: number, y: number, z: number) {
    const m = this.viewProjection();
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const r = (this.app.graphicsDevice.canvas as HTMLCanvasElement).getBoundingClientRect();
    return {
      x: (cx / w) * r.width * 0.5 + r.width * 0.5,
      y: r.height * 0.5 - (cy / w) * r.height * 0.5,
      depth: w,
      onScreen: w > 0
    };
  }

  /** Camera-space depth of a world point, for depth-limited brushing. */
  viewDepth(p: pc.Vec3): number {
    const fwd = this.camera.forward;
    const eye = this.camera.getPosition();
    return (p.x - eye.x) * fwd.x + (p.y - eye.y) * fwd.y + (p.z - eye.z) * fwd.z;
  }
}
