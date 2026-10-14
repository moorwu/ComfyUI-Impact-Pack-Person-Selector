[English](README.md) | **简体中文** | [日本語](README.ja.md)

# ComfyUI-Impact-Pack-Person-Selector

[ltdrdata/ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) 的一个分支，新增了 **Person 节点**：从群像照片里挑出指定的人，只重绘他们，而不是把画面里每个人都精修一遍。可以按编号挑、按性别挑、按角色名挑，也可以用大白话描述，比如"穿牛仔外套的那个男的"。

除此之外，这个节点包与上游完全一致。上游原始说明（英文）见 [README.md](README.md)。

## 这个分支加了什么

  * **只重绘选中的人。** `Person Detector (SEGS)` 给画面里每个人编号，`Person Selector` 留下你要的那些，把他们的区域作为 SEGS 输出，交给包里原有的 `Detailer (SEGS)`。
  * **四种挑人方式。** 编号（`2`、`1,3`、`2-4`、`-1` 表示最后一个）、性别、角色名、自由文本描述 —— 而且可以组合使用。
  * **脸或全身。** `face_SEGS` 和 `person_SEGS` 两个独立输出，既可以只精修脸，也可以改一个人的衣服、发型和姿态。
  * **不用额外下载 VLM。** 角色名和描述的匹配，跑在 Krea 2 工作流已经加载的 Qwen3-VL 文本编码器上。
  * **为真实群像设计。** 能应付前后景深不同、身体互相遮挡、脸被挡住一半的情况：背景路人和糊掉的脸会被过滤，编号锚点从脸退到头、再退到身体，并且每个人的蒙版会挖掉其他已编号者的脸，重绘时不会糊到旁边的人身上。
  * **可选的精细化。** 实例分割（`SEGM_DETECTOR`）或 SAM 让身体蒙版贴合轮廓；针对插画风格的图，还可以接动漫专用的人体/头部检测模型。

## 前置要求

  * [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack)，用于提供 `UltralyticsDetectorProvider`；并在 `models/ultralytics/` 下准备 `segm/person_yolov8m-seg.pt` 和 `bbox/face_yolov8m.pt` 两个模型。
  * 可选，使用 `gender` / `description` / 角色名时需要：Krea 2 文本编码器（`CLIPLoader` 的 `type` 选 `krea2`，例如 `qwen3vl_4b_fp8_scaled.safetensors`）。
  * 可选，处理动漫和插画时推荐：[deepghs](https://huggingface.co/deepghs) 的 `bbox/person_detect_v1.1_m.pt` 和 `bbox/head_detect_v2.0_s_yv11.pt`（MIT 协议）。需要把文件名加进 `ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt`。
  * 可选，需要贴合轮廓的身体蒙版时：任意 SAM 模型，通过 `SAMLoader` 加载（例如 `sams/sam_vit_b_01ec64.pth`）。

安装方式与上游相同，见 [How To Install](README.md#how-to-install)，把地址换成本仓库即可。请用它**替换**上游那个包，不要两个同时装 —— 二者注册的节点名相同，ComfyUI 只会加载其中一个。上游说明里写的目录名是 `ComfyUI-Impact-Pack`，在这个 fork 下就是你克隆本仓库时用的目录名。

## 模型下载

| 模型 | 放在哪里 | 下载地址 |
| --- | --- | --- |
| `person_yolov8m-seg.pt` | `ComfyUI/models/ultralytics/segm/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt) |
| `face_yolov8m.pt` | `ComfyUI/models/ultralytics/bbox/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt) |
| `person_detect_v1.1_m.pt`（可选，动漫用） | `ComfyUI/models/ultralytics/bbox/` | [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.1_m/model.pt)（MIT） |
| `head_detect_v2.0_s_yv11.pt`（可选，动漫用） | `ComfyUI/models/ultralytics/bbox/` | [deepghs/anime_head_detection](https://huggingface.co/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_s_yv11/model.pt)（MIT） |
| `sam_vit_b_01ec64.pth`（可选） | `ComfyUI/models/sams/` | [segment-anything](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| `qwen3vl_4b_fp8_scaled.safetensors`（可选，用 `gender` / `description` 时必需） | `ComfyUI/models/text_encoders/` | Krea 2 的文本编码器 —— 如果你本来就在跑 Krea 2 工作流，这个文件已经有了 |

deepghs 的两个模型在 Hugging Face 上都叫 `model.pt`、各自放在版本目录里，所以**下载时必须改名**。另外这两个文件名还要加进 Impact-Subpack 的白名单，`UltralyticsDetectorProvider` 才会列出它们：

```bash
cd ComfyUI/models/ultralytics/bbox
curl -L -o person_detect_v1.1_m.pt https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.1_m/model.pt
curl -L -o head_detect_v2.0_s_yv11.pt https://huggingface.co/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_s_yv11/model.pt
printf 'person_detect_v1.1_m.pt\nhead_detect_v2.0_s_yv11.pt\n' >> ../../../user/default/ComfyUI-Impact-Subpack/model-whitelist.txt
```

加完模型后重启 ComfyUI，加载器才会认到。

## 快速上手

加载 `example_workflows/person_detailer.json`。它已经连好了下面这张图，并且同时演示了"重绘脸"和"重绘全身"两条分支：

```
UltralyticsDetectorProvider (person) ─┐
UltralyticsDetectorProvider (face) ───┼─> Person Detector (SEGS) ─persons─> Person Selector ─face_SEGS──> Detailer (SEGS)
LoadImage ────────────────────────────┘                    (same image) ─┘  └person_SEGS─> Detailer (SEGS)
```

  1. 把**同一张图**分别接到 `Person Detector (SEGS)` 和 `Person Selector` —— Selector 会核对尺寸，不一致会直接报错。
  2. 在 `Person Selector` 上说明你要谁：`index`（例如 `2`）、`gender`，以及/或者 `description`（例如 `the woman in the red apron`）。全部留空则表示选所有人。
  3. **只重绘脸**：把 `face_SEGS` 接到 `Detailer (SEGS)` 的 `segs` 输入。**重绘整个人**（衣服、姿态、发型、身体）：改接 `person_SEGS`。重绘全身时建议在 `Person Detector (SEGS)` 上接 `person_segm_detector` 或 `sam_model`，蒙版才会贴合人物轮廓而不是一个方框；`denoise` 取 0.5–0.6 时衣服变化明显，但选中者自己的脸也会跟着变，要保住脸就在之后单独再跑一遍脸部重绘。
  4. 两个节点的 `preview` 输出可以看到编号和蒙版；Selector 的 `debug_text` 会给出检测/排除的数量，以及 VLM 的回答内容。

## 示例

下面每张图都由 `example_workflows/person_detailer.json` 跑 `tests/person_fixtures_v2/` 里的测试图得到，这些测试图随仓库一起提供，你可以自己复现。

**1. 检测与编号** —— `Person Detector (SEGS)` 的 `preview` 输出。这张错落站位的合影里找到 9 个人，每人一张按其编号配色的全身蒙版，编号标在脸的上方。

![检测预览：9 个人各有带色蒙版和编号](docs/images/detect-preview.jpg)

**2. 选人** —— `Person Selector` 的 `preview` 输出，`description` 填的是 *the woman in the red graduation gown holding a bouquet*（穿红色学士袍、抱着花束的女士）。VLM 选中了 2 号（绿色框），其余人保持灰色、原样输出。

![选人预览：2 号被绿框标出，其余为灰色](docs/images/selection-preview.jpg)

**3. 重绘脸** —— `face_SEGS` 接 `Detailer (SEGS)`，提示词为 *close-up portrait of a face with bright blue eyes and a big open smile, detailed skin*（蓝眼睛、灿烂笑容的面部特写）。只有 2 号的脸被重采样，另外 8 人保持不变。

| 重绘前 | 重绘后 |
| --- | --- |
| ![原始合影](docs/images/face-before.jpg) | ![只有被选中女士的脸被重绘](docs/images/face-after.jpg) |

**4. 重绘全身** —— `person_SEGS` 接 `Detailer (SEGS)`，提示词为 *a person wearing a bright pink outfit, detailed clothing*（穿亮粉色衣服），`denoise` 取 0.6。只有被选中男士的衣着变了，他身旁两人和背景里的咖啡师都没有受影响。

| 重绘前 | 重绘后 |
| --- | --- |
| ![原始咖啡馆照片](docs/images/body-before.jpg) | ![被选中的男士牛仔外套里换成了粉色上衣](docs/images/body-after.jpg) |

**5. 什么情况下会翻车** —— 同样的脸部重绘、与示例 3 相同的设置，但换成 4 号：她是侧脸，而且被前面的女士挡住了一部分。detailer 把这块裁剪重建成了一只正面的大眼睛 —— 提示词里写了 *bright blue eyes*（明亮的蓝眼睛），而原图几乎没有可依托的面部信息。

| 重绘前 | 重绘后 |
| --- | --- |
| ![戴红头巾的女士，侧脸](docs/images/limit-profile-before.jpg) | ![她的脸被重建成了一只正面的大眼睛](docs/images/limit-profile-after.jpg) |

侧脸、被严重遮挡、或者只有几十像素大小的脸，是这套节点的薄弱环节 —— 对原版 `FaceDetailer` 同样如此。应对办法：调低 `denoise`、不要在提示词里要求裁剪区域根本看不到的特征，或者干脆不选这些人。

## 人物重叠：本次修复与限制

身体蒙版现在会把共享像素分给唯一的已编号人物，分空后保持为空；脸与身体的关联优先看轮廓覆盖，再看框的重合。SAM 候选只选一个，不再合并多个备选。这些改动能避免节点重新引入共享蒙版，但几何分配不能还原交叉手臂或腿部的真实边界。

一张下载的拥抱照片中，`person_yolov8m-seg.pt` 对两个已检测到的脸只给出一个身体实例。因此，接上 `person_segm_detector` 不代表一定能把两个人分开，SAM 1 细化也未解决这个例子。我们另外测试了 [SAM 3](https://github.com/facebookresearch/sam3)，分别使用 `man`、`woman` 文本提示：下面两张照片都得到两份独立的可见人物轮廓。脚本把这些蒙版送入实际 Detector → Selector 节点，验证共享像素为 0，导出的蒙版保持一致。这是分割实验，尚未接成内置 SAM 3 加载节点，也没有运行 Detailer 重绘；不能照这个例子通过现有 `SAMLoader` 接入 SAM 3。

从左到右：原图、Detector 预览、人物 1、人物 2。被遮挡的身体部分不会补全；第二张仍有少量边缘碎片，尚未做像素级真值精度评测。

![拥抱照片：SAM 3 分离弯曲腿部和交叉手臂](docs/images/person-overlap-sam3-first.jpg)

![拥抱照片：只有一个可见脸，仍输出两份身体蒙版](docs/images/person-overlap-sam3-second.jpg)

图片来源：[第一张](https://images.unsplash.com/photo-1575216525169-112eb232f906?auto=format&fit=max&w=1600&q=90)、[第二张](https://images.unsplash.com/photo-1741217531361-b9f5ecbb6fe2?auto=format&fit=max&w=1600&q=90)，于 2026-10-13 从 Unsplash 下载，与上面的仓库测试图示例分别验证。

复现时，按官方仓库准备 SAM 3 源码、`sam3.pt` 权重及依赖，并准备带有 `models/ultralytics/bbox/face_yolov8m.pt` 的 ComfyUI 环境。在本仓库根目录运行：

```bash
python tests/person_e2e/sam3_person_check.py \
  --image /path/to/photo.jpg --checkpoint /path/to/sam3.pt \
  --sam3-root /path/to/sam3 --comfy-root /path/to/ComfyUI \
  --output /path/to/results --device cpu --prompts man woman
```

[验证脚本](tests/person_e2e/sam3_person_check.py) 会保存独立蒙版、抠图、对比图和 `report.json`。CPU 模式只在隔离的 SAM 3 源码副本中应用三处兼容处理，不修改已安装的包。显存充足时可用 `--device cuda`。这两个文本提示针对本次照片，其他场景需要调整描述并检查分割结果。

## Person 节点说明

  * `Person Detector (SEGS)` —— 检测画面里所有人和所有脸，把每张脸和它所属的身体关联起来（找不到身体时，会为这张脸合成一个近似的身体框），排除太小的、太靠后景的、太模糊的人，剩下的从左到右编号（可通过 `sort_by` 调整）。
    * 必填：`image`、`person_detector` 和 `face_detector`（都是 `BBOX_DETECTOR`，例如用 `UltralyticsDetectorProvider` 加载 `segm/person_yolov8m-seg.pt` 和 `bbox/face_yolov8m.pt`）。所有接上的检测器（身体、SEGM、额外人体、头部、脸）共用同一个 `threshold`。
    * 可选 `person_segm_detector`（`SEGM_DETECTOR`，例如与 `person_detector` 同一个 `UltralyticsDetectorProvider` 的 `SEGM_DETECTOR` 输出）：接上后身体检测切换为实例分割，每个人的全身区域是真实轮廓而不是矩形；它会取代 `person_detector`，后者将被忽略。
    * 可选 `extra_person_detector`（`BBOX_DETECTOR`）和 `head_detector`（`BBOX_DETECTOR`）：`extra_person_detector` 是优先运行的额外人体检测器（动漫/插画推荐 `bbox/person_detect_v1.1_m.pt`），`person_segm_detector`/`person_detector` 的框只用来补它漏掉的人。`head_detector`（推荐 `bbox/head_detect_v2.0_s_yv11.pt`）检测头部，为看不到脸的人提供编号锚点、预览标签位置和 SAM 提示点；接上之后，`min_relative_size` 的比较对象也会随之改变。两个模型都来自 Hugging Face 用户 [deepghs](https://huggingface.co/deepghs)（MIT 协议）—— 下载到 `models/ultralytics/bbox/`，并把文件名加进 `ComfyUI/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt`，`UltralyticsDetectorProvider` 才会加载它们。
    * 可选 `sam_model`（`SAM_MODEL`）：用此人的脸（看不到脸时用头）作为正向点，其他已编号者的脸/头作为负向点，细化身体蒙版。邻居的头即使在身体框外，其肢体也可能交叠，因此仍提供负向点；落在自己的脸/头区域内的负向点会跳过。SAM 的多个候选不会合并：选取满足全部正负提示的最大候选，并裁到身体框内；没有合格候选时使用检测器蒙版。
    * 排除阈值与默认值：`min_person_ratio` 0.015（身体框面积占全图的比例）、`min_relative_size` 0.31（过滤背景路人 —— 逐人判定：该人检测到头就用头的短边比最大的头，否则用脸的短边比最大的脸，再否则用身体面积比最大者的平方根）、`min_face_size` 24 像素（脸框短边）、`min_sharpness` 0 表示关闭（脸部裁剪的拉普拉斯方差，用于过滤模糊的背景人物）。
    * 存过的 v1 工作流会沿用旧的控件值（`min_person_ratio` 0.02、`min_relative_size` 0.25）；由于尺寸度量改成了线性方式，请改为 0.015 / 0.31。
    * 输出 `persons`（`PERSONS`，接给 `Person Selector`）和 `preview`（`IMAGE`）：每人的身体蒙版按编号配色，标签放在脸/头上方。蒙版先挖掉邻居外扩 15% 的脸/头区域；两个检测轮廓之间只挖除邻居蒙版也覆盖的像素，保留自己的头发和手臂。剩余身体交叠像素分给唯一的人：优先保留自己的脸/头、优先检测到的身体而非合成矩形，其余按最近的脸/头中心（没有时用蒙版中心）分配。脸与身体的关联优先看轮廓覆盖，再看框的重合。身体蒙版分空后保持为空，脸仍可单独选择。只有矩形时，这种几何分配无法还原真实遮挡边界。被排除的人不参与挖除，显示为灰框，并标注 `xS`（太小）、`xBG`（背景）、`xF`（脸太小）或 `xB`（太模糊）。
    * 编号方式（`sort_by`）：默认 `left_to_right` 按脸的中心排序，没有脸时退回头的中心，两者都没有时用身体框的中心。
  * `Person Selector` —— 从 `Person Detector (SEGS)` 编号过的人里选出一部分，把他们的脸和身体区域作为 SEGS 输出，可直接接 `Detailer (SEGS)`。
    * `index`：从 1 开始、用逗号分隔的编号，例如 `2`、`1,3`、`2-4`，或用 `-1` 表示最后一个人；留空表示选所有人。
    * `gender`（`any`/`male`/`female`）和 `description`（自由文本，虚构角色可以直接写角色名 —— 真实人物只按描述匹配，不做身份识别）需要在 `clip` 上接 Krea 2 文本编码器（`CLIPLoader` 的 `type` 设为 `krea2`，例如 `qwen3vl_4b_fp8_scaled.safetensors`）；节点会在图上画出带编号的框，让 VLM 挑出符合条件的编号。可选的 `verify` 会把每个挑中的人单独裁出来复核 —— 其他人的身体蒙版涂成灰色，确保 VLM 只看被挑中的这个人 —— 只保留得分高于 `verify_threshold` 的结果。
    * `image` 必须与接给 `Person Detector (SEGS)` 的是同一张图 —— 节点会核对尺寸，不一致就报错。
    * 输出 `face_SEGS`/`person_SEGS`（选中的）和 `remained_face_SEGS`/`remained_person_SEGS`（已编号但未被选中的），一个 `preview` 图像，以及一个 `debug_text` 字符串，内含检测/排除的数量和 VLM 的推理过程。
    * 注意事项：被 `Person Detector (SEGS)` 排除的人，既不在选中的 SEGS 里，也不在 remained 的 SEGS 里 —— 他们被完全丢弃。当某个选中的人没有检测到脸时，`face_SEGS` 和 `person_SEGS` 的位置不是一一对应的（这个人只是不出现在 `face_SEGS` 里）。
