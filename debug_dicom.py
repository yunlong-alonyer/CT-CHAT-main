import os
import pydicom

# 替换成你报错的真实路径
dicom_dir = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/8383204530008"

# 1. 检查是否读取到了文件
files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
print(f"[*] 扫描到文件数量: {len(files)}")

if not files:
    print("[!] 确实没有找到任何普通文件。")
    exit()

# 2. 拿前 3 个文件开刀，打印最底层的报错
print("\n[*] 正在尝试暴力解析前 3 个文件...")
for i, f in enumerate(files[:3]):
    print(f"\n--- 测试文件 {i + 1} ---")
    try:
        ds = pydicom.dcmread(f, force=True)
        print("  - [成功] pydicom.dcmread 解析通过！")

        # 很多无后缀文件丢失了元数据，强行补全
        if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            print("  - [警告] 缺少传输语法，已强行补全 LittleEndian")

        # 尝试提取图像矩阵 (最容易崩的一步)
        img_array = ds.pixel_array
        print(f"  - [成功] 像素矩阵提取成功！形状: {img_array.shape}")

    except Exception as e:
        print(f"  - [致命错误] 解析失败，原因: {type(e).__name__}: {e}")