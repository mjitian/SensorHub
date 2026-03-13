# 天工2.0 Pro (Tiangong 2.0 Pro) 部署指南

## 概述

本文档说明如何将微调后的 XR-1 模型与 SensorHub 库配合使用，在天工2.0 Pro (Tiangong 2.0 Pro) 平台上进行部署。

---

## 常见问题解答

### Q1：微调了 XR-1 模型，但微调数据是天工的，模型的输出会自动匹配天工的数据吗？

**结论：不会自动匹配，需要确保观测空间和动作空间的维度与配置一致。**

详细说明：

- 使用天工数据微调 XR-1 模型时，模型的输入输出维度由**训练时的数据格式**决定，而不是由底层模型架构自动推断。
- 若微调所用天工数据的关节维度、摄像头配置与 XR-1 的原始训练配置**相同**，则模型输出可与天工数据格式兼容。
- 若两者的动作空间（action space）或观测空间（observation space）存在差异（例如关节数量、顺序不同），则需要在推理阶段进行**适配映射**，否则会导致关节控制错误。

**建议检查以下内容**：

| 检查项 | XR-1 默认配置 | 天工2.0 Pro 配置 | 是否一致 |
|--------|--------------|-----------------|----------|
| 动作向量维度 | 26-dim | 视具体配置而定 | 需人工核对 |
| 左臂关节数 | 7 (indices 0–6) | 视具体配置而定 | 需人工核对 |
| 左手关节数 | 6 (indices 7–12) | 视具体配置而定 | 需人工核对 |
| 右臂关节数 | 7 (indices 13–19) | 视具体配置而定 | 需人工核对 |
| 右手关节数 | 6 (indices 20–25) | 视具体配置而定 | 需人工核对 |
| 摄像头数量与分辨率 | 1×(640×360) | 视具体配置而定 | 需人工核对 |

如存在差异，请参阅下方的「动作空间适配」章节。

---

### Q2：在哪里部署？

**部署位置：搭载 GPU 的机载计算机（Onboard Computer），通过 ROS2 与机器人硬件通信。**

推荐部署架构如下：

```
[天工2.0 Pro 机器人本体]
       │
       ├── 嵌入式控制器（Arduino / ESP32）
       │       └── SensorHub 库 (I2C/I2S 传感器读取)
       │               └── 传感器数据 ──► ROS2 Topic
       │
       └── 机载计算机（NVIDIA Jetson / x86 with GPU）
               ├── ROS2 环境
               ├── 摄像头驱动节点
               ├── 关节状态节点
               └── 策略推理节点 (ros2_deployment_HAND.py)
                       └── 输出关节指令 ──► 机器人执行器
```

---

## 部署步骤

### 1. 环境准备

在机载计算机（或外部工作站）上安装依赖：

```bash
# 安装 ROS2 (推荐 Humble 或 Jazzy)
# 参考：https://docs.ros.org/en/humble/Installation.html

# 安装 Python 依赖
pip install torch torchvision opencv-python numpy

# 安装 lerobot
pip install lerobot
```

### 2. 配置模型路径

参考 [x-humanoid-training-toolchain/deployment](https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain/tree/main/deployment)，编辑部署脚本中的模型路径：

```python
# 在 ros2_deployment_HAND.py 中设置您的微调模型路径
model_path = "/path/to/your/finetuned_xr1_tiangong_model"
```

### 3. 启动 ROS2 节点

```bash
# 启动摄像头驱动节点
ros2 launch your_camera_package camera.launch.py

# 启动关节状态发布节点
ros2 launch your_robot_description joint_state_publisher.launch.py
```

### 4. 运行策略推理

```bash
python ros2_deployment_HAND.py
```

### 5. 验证关节动作映射

确认 `publish_action` 中的关节切片顺序与天工2.0 Pro 的硬件配置一致：

```python
def publish_action(self, action):
    # 按照 26-dim 动作向量的标准顺序切片
    target_joint = np.concatenate([action[:7], action[13:20]])  # 左臂7 + 右臂7
    left_hand_pos = action[7:13]                                # 左手6
    right_hand_pos = action[20:26]                              # 右手6
```

---

## 动作空间适配

若天工2.0 Pro 的关节配置与 XR-1 默认 26-dim 动作空间不同，需要添加适配层：

```python
def adapt_action_for_tiangong(action, xr1_to_tiangong_mapping):
    """
    将 XR-1 模型输出的动作向量映射到天工2.0 Pro 的关节顺序。

    Args:
        action (np.ndarray): XR-1 模型输出的原始动作向量 (26-dim)
        xr1_to_tiangong_mapping (list): 索引映射列表，例如 [0, 1, 2, ...]

    Returns:
        np.ndarray: 适配后的动作向量
    """
    adapted_action = np.zeros(len(xr1_to_tiangong_mapping))
    for tg_idx, xr1_idx in enumerate(xr1_to_tiangong_mapping):
        adapted_action[tg_idx] = action[xr1_idx]
    return adapted_action
```

---

## SensorHub 与部署流程的集成

SensorHub 库在嵌入式端负责通过 I2C 读取机器人传感器数据，并将其作为 ROS2 话题发布，供策略推理节点使用：

```cpp
#include <SensorHub.h>

// 假设关节编码器连接在 I2C 总线上，地址为 0x40
SensorHub jointSensor(0x40);

void loop() {
    uint16_t jointAngle;
    if (jointSensor.i2c_read_Xbit_LE(0x00, &jointAngle, 16)) {
        // 将关节角度发布至 ROS2 串口桥节点
        Serial.print("JOINT:");
        Serial.println(jointAngle);
    }
    delay(10); // 100 Hz 采样率
}
```

---

## 注意事项

1. **微调数据格式验证**：在部署前，务必使用离线脚本验证微调模型的输入输出格式与实际硬件配置是否一致。
2. **安全性**：初次部署时建议在仿真环境（如 Isaac Sim 或 Gazebo）中验证策略正确性，再上线实体机器人。
3. **实时性**：策略推理应在具有 GPU 的机载计算机上运行，以确保满足实时控制要求（推荐推理延迟 < 50ms）。
4. **日志记录**：建议在部署时开启 ROS2 bag 录制，以便事后分析异常行为。

---

## 参考资源

- [x-humanoid-training-toolchain 部署代码](https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain/tree/main/deployment)
- [ROS2 官方文档](https://docs.ros.org/en/humble/)
- [SensorHub 库文档](../Readme.md)
