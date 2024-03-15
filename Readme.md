- download https://github.com/comfyanonymous/ComfyUI
- 秋叶整合包 https://www.bilibili.com/video/BV1Ew411776J/?spm_id_from=333.999.0.0&vd_source=2774965e63414d32c855e45d7fca856a
- OOTD duffsion https://openart.ai/workflows/datou/ootdiffusion/8XE70w17xstgLBOCl4Bl
- 

# 基础操作

https://www.comflowy.com/zh-CN/basics/basic-nodes

+ 拖拽多个组件

  按住 ctrl 多选, 按住 shift 拖动

+ 带连线复制 ctrl + shift + v 

+ 让线更好看 utils - Reroute Primitive 

+ undo: ctrl+z

+ redo: ctrl+r

+ save: shift + s  

# 入门

https://www.youtube.com/watch?v=SPCaY1Q-P88&list=PLre94Fo0ReBLqf3yoYlnxU-dqKCg-sXTM&index=3

## 加载模型 Module

add node - loader - load checkpoint

## 提示词 Prompt

conditioning - clip text encoder (Prompt)

## 分组 Group

add color, add group, add general-note

## 采样器 Sampler

+ 从 clip text encoder (Prompt) CONDITIONING 拉一根线出来, KSampler
+ 或者 sampling - Ksampler

> 参数说明: 
>
> + seed: 采样器编号, fixed/randomize, 同一张图片, 同一个seed
> + add_noise: enable
> + cfg: 与 prompt 相关程度, 建议在 10 左右
> + sampler_name: 由经典的常微分方程求解方程命名的sampler
>   + ancestral/sde: 每一step 添加随机采样, 不稳定, 随机性更强
>   + uni_pc: 适用于 10 step左右, 出现较好的效果
>   + 推荐: euler_ancestral, dpmpp_2m, dpmpp_2m_sde_gpu, uni_pc
> + scheduler: 在每一步采样器的大小
>   + normal: 线性
>   + karras: 平滑
>   + 后四个不推荐
> + denoise: 降噪百分比, 1 代表 100% steps
> + return_with_leftover_noise: 将剩余步数传递给下一个采样器
> + start_at_step/end_at_step: 与上一个采样器相关联

## 潜在空间图片 LATENT

选择空图片, 指定尺寸即可 

## 生成噪声图

+ save
+ preview

# Refiner 模型

![Screenshot 2024-02-27 at 16.27.52](/Users/vjf/Library/Application Support/typora-user-images/Screenshot 2024-02-27 at 16.27.52.png)

base模型生成小图, refiner模型生成大图  

+ base修改: 

  + add note base
  + change CLIP Text Encode to input
  + text 拉出来,  add node - utils - Primitive 
  + 采样器改为 sampling - KSampler(Advanced)

+ refiner:

  + 增加 refiner - checkpoint
  + 复制 prompt , text 连到 base text
  + base 高阶采样器输出到 refiner 高阶采样器, steps 与 base 一致, start_at_step = base end_at_step, end_at_stetp >= steps, return_with_leftover_noise false(可以使用 utils-Primitive 简化)

# Rave & animateDiff
https://www.youtube.com/watch?v=7ZxsBmUm3Lg
+ workflow: https://github.com/Nuked88/DreamingAI/blob/main/T13_video_to_video.json
+ customNode
  + ComfyUI-N-Suite
  + comfyui-reactor-node
+ model
  + Checkpoint: [MeinaMix_V10](https://huggingface.co/Meina/MeinaMix_V10/tree/main/safety_checker)
  + Controlnet: [control_v11p_sd15_lineart.pth](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main)
  + Lora: [pytorch_lora_weights.safetensors](https://huggingface.co/Yashhhhmishra/pytorch_lora_weights.safetensors/blob/main/pytorch_lora_weights.safetensors)
  + animatediff_models: [temporaldiff-v1-animatediff.ckpt](https://huggingface.co/CiaraRowles/TemporalDiff/blob/main/temporaldiff-v1-animatediff.ckpt)
  + ReActorFaceSwap: 
    + insightface: [inswapper_128.onnx](https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main)
    + facerestore_models: [codeformer.pth](https://github.com/sczhou/CodeFormer/releases)

+ Pip requirements
  + imageio-ffmpeg
![1710498296388_919E60E5-10D8-46fa-B10B-3CD98A7DC0DB.png](pic%2F1710498296388_919E60E5-10D8-46fa-B10B-3CD98A7DC0DB.png)

# 一些有意思的项目

+ https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM
+ https://openart.ai/workflows/datou/ootdiffusion/8XE70w17xstgLBOCl4Bl

# 人像检测

## 使用 comfyUI manager 安装 custom node

+ open comfyUI manager
+ install Custom Nodes

## 使用 yolov8 实现人脸检测

https://github.com/zcfrank1st/Comfyui-Yolov8

+ yolov8 - detection/seg
+ mask - convert mask to image

## yolo-world
+ https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM
+ https://github.com/AILab-CVC/YOLO-World
+ https://github.com/yformer/EfficientSAM

# 其他

## 模型
+ https://civitai.com/
+ https://www.bilibili.com/read/cv23887580/
+ https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending&search=stable-diffusion

## works 
+ https://comfyanonymous.github.io/ComfyUI_examples/
+ https://comfyworkflows.com/
+ https://openart.ai/workflows

## prompt 
+ https://prompthero.com/midjourney-prompts

## 一些名词
+ SDXL
+ RAVE: https://rave-video.github.io/
+ animateDiff
  + https://www.youtube.com/watch?v=fxNqWTGh3ZM
  + https://www.youtube.com/watch?v=D3A6GeYq5LQ
  + https://heehel.com/aigc/comfyui-animationdiff.html
+ heygen: 数字人做的比较好的国外厂商/ ai avatar video
+ dreamTalk: github
+ wav2lip: github
+ sadTalk: github
+ tts
+ sag
+ sam
+ controller
+ lora
+ sora
+ rag