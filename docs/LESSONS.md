# Lessons learned (verified by rendering, Blender 5.2.2 on an M2)

Kept from the earlier development stages, whose reports were retired. Each item cost real debugging time; the code relies on all of them.

## Blender / Cycles behaviour
- **Object Index pass is unfiltered.** Each pixel holds exactly one object's `pass_index`, so masks have no anti-aliasing mixes. `view_layer.pass_alpha_threshold = 0.5` makes alpha-cut leaves count only where opaque.
- **`Object.pass_index` is capped at 32767.** Larger values are silently clamped. Mass-scattered "stuff" therefore shares one id per kind; countable objects get unique ids (`configs/classes.json → instance_kinds`). The registry raises instead of clamping.
- **Instancing and ids.** GN *geometry* reports the owner object's `pass_index`; *object references* (Collection Info) keep the referenced object's own index. Trees use Object Info geometry wrapped as instances (shared memory, per-tree id). Scatter prototypes carry their kind's shared id.
- **AOVs add up through transparent surfaces** (leaf cards). Don't use AOV labels without a binary-alpha label mode.
- **GN modifier inputs in 5.2:** set them with `modifier.properties.inputs.<Socket_N>.value`. The old `md["Socket_N"]` is rejected.
- **The 5.2 Compare node** only exposes sockets of its active data type; select sockets by name and type, not by index.
- **Objects in excluded collections are not evaluated** when referenced "as instance". Reference them as geometry.
- **Metal backend:** scene-specialised kernel compilation crashed (nil NSURL) during long renders, so `kernel_optimization_level = OFF`. A rare abort still occurs, so `generate.blender_build` retries once on a signal exit.
- **Render cost** is dominated by instance count, not samples. One scanned pine has ~27k twig instances; thinning them to 30 % and enlarging the kept ones looks identical from the air and cuts a 1-sample render from 44 s to 12 s.
- **Sky texture `sun_rotation` equals the compass azimuth** (0 = north/+Y, 90° = east/+X).
- **The Hue/Saturation node's hue 0.5 means no change.**
- **zsh does not split `$var` into words** in `for` loops: `set -- $ps` silently passes one argument. Loop over explicit lists.
- **Coplanar duplicate faces render black** in Cycles (each shadows the other). Never emit two faces at the same height over the same area.
- **Noise Texture `Scale` is not a seed.** Passing a "seed" as scale shrank layer noise to sub-pixel size, where it averages to 0.5 and does nothing. Use `NodeBuilder.noise_m(uv, wavelength_m, seed)`, which offsets the pattern instead.
- **Noise Texture output stays near 0.5** (roughly ±0.1). Stretch it with Map Range before using it as an amplitude.

## Python / IO
- **OpenImageIO `write_image` scrambles 2-D arrays.** Write masks as `(h, w, 1)`. `annotations.write_png` reads every file back and compares.
- **macOS `unzip` fails on several Megascans zips.** Python's `zipfile` reads them.
- **iCloud-synced folders** block reads and upload everything. Heavy data lives in `~/synth_aerial_work` (`SYNTH_WORK_ROOT`).
- **zsh doesn't word-split unquoted variables** in shell loops.

## Realism
- **Scanned textures need checking by eye**:
  - "dusty rough asphalt" is pink-beige;
  - "roof tiles" are grey slate;
  - "stucco" is orange;
  - "wild grass" is dark green;
  - "stone wall" is cool grey.
  
  Roles and colour corrections live in the configs.
- **Measure colours against real imagery** (classify bright pixels into beige / grey / white / red and compare shares and means), rather than judging by eye. Arab-town roofs in the Galilee are mostly warm beige, not neutral grey.
- **Flat roofs must read lighter than the dry ground** from the air (light grey or white coating). The "dusty concrete floor" scan rendered darker than straw until it was brightened.
- **Measure, don't eyeball, layer colours:** render each ground layer alone at aerial GSD (`synth/blender/swatches.py`) and calibrate its HSV correction to a target colour (`synth.steps.step3 --calibrate`). Before calibration, every layer rendered as the same dull grey-brown.
- **Dry season:** bright straw grass dominates and darker soil shows through. The inverse looks like camouflage.
- **Roads:**
  - choose endpoints the grade limit can actually connect;
  - start side roads perpendicular to their parent;
  - clamp grade forward from the junction;
  - keep the carriageway above the ground everywhere;
  - forbid crossings.
- **Terrain rings join without seams** only if the inner ring has an exact hole, the core has a separate skirt, slope for land cover comes from one common grid, and micro-relief tapers at the core edge.
- **Pick a site that can host the village**, and fall back to the next ranked site otherwise.
