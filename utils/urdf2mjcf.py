import os
from dm_control import mjcf

#####   去/.mujoco下面跑./compile
# 读取 URDF
path = '/home/island/Desktop/mobile_manipulation/realrobot/moma_mujoco/assets/island_moma_robot/urdf'
urdf_path = os.path.join(path, 'rm_65_6f_description.urdf')  # 替换为你的 URDF 文件路径
mjcf_model = mjcf.from_urdf(urdf_path)

# 输出转换后的 XML
mjcf_xml = mjcf_model.to_xml_string()

# 保存到文件
with open("output.xml", "w") as f:
    f.write(mjcf_xml)

print("转换完成，MJCF 文件已保存为 output.xml")