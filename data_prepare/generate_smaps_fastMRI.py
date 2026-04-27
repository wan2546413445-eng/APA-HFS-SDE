import sys, os
'''
BART_ABS_PATH = "/home/wanshiyu/PycharmProjects/HFS-SDE-master/bart"

# 1. 将BART的python目录加入Python搜索路径（必须是绝对路径）
sys.path.insert(0, os.path.join(BART_ABS_PATH, "python"))
# 2. 设置TOOLBOX_PATH为BART根目录（指向可执行文件）
os.environ["TOOLBOX_PATH"] = BART_ABS_PATH
# 3. 限制线程数（保留原逻辑）
os.environ["OMP_NUM_THREADS"] = "1"
'''
import glob
import numpy as np
import h5py
import sigpy as sp
# ===================== 核心修复：完整配置BART Python环境 =====================
# 替换为你的BART绝对路径！
BART_ABS_PATH = "/mnt/SSD/wsy/bart"
BART_PYTHON_DIR = os.path.join(BART_ABS_PATH, "python")

# 1. 验证BART python目录和核心文件是否存在
assert os.path.exists(BART_PYTHON_DIR), f"❌ BART python目录不存在：{BART_PYTHON_DIR}"
assert os.path.exists(os.path.join(BART_PYTHON_DIR, "bart.py")), "❌ 缺少bart.py"
assert os.path.exists(os.path.join(BART_PYTHON_DIR, "cfl.py")), "❌ 缺少cfl.py"

# 2. 将BART python目录加入sys.path（让Python能找到所有依赖）
if BART_PYTHON_DIR not in sys.path:
    sys.path.insert(0, BART_PYTHON_DIR)

# 3. 配置BART运行环境
os.environ["TOOLBOX_PATH"] = BART_ABS_PATH  # 指向BART可执行文件目录
os.environ["OMP_NUM_THREADS"] = "1"         # 限制线程数
os.environ["PATH"] += os.pathsep + BART_ABS_PATH  # 确保能找到bart可执行文件

# 4. 导入BART（此时能找到cfl等依赖）
try:
    from bart import bart
    print("✅ BART Python接口（含cfl依赖）导入成功！")
except Exception as e:
    print(f"❌ 导入失败：{e}")
    sys.exit(1)
#print('loadBart')

#output_dir = '.'
#input_dir = '/data0/chentao/data/fastMRI_knee_test/T1_data'
input_dir = '/mnt/public/成像组/dataset/fast_MRI/multicoil_brain/brain_multicoil_train_batch_0/multicoil_train'  # kspace文件目录
output_dir = '/mnt/SSD/wsy/fastmri_data/brain_multicoil_train/maps'  # maps目录保存灵敏度图

def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted(glob.glob(input_dir + '/*.h5'))

    # 先测试前 3 个，确认没问题后再注释掉这一行
    #file_list = file_list[:1]

    for file in file_list:
        print('*********Load next MRI Data************')
        # Load specific slice from specific scan
        basename = os.path.basename( file )
        output_name = os.path.join( output_dir, basename )
        if os.path.exists( output_name ):
            print("skip existing:", output_name)
            continue
        with h5py.File(file, 'r') as data:
            #num_slices = int(data.attrs['num_slices']) 368x302x256x18
            kspace = np.array( data['kspace'] ) # 40x15x640x368
            print('kspace shape:',kspace.shape)
            s_maps = np.zeros( kspace.shape, dtype = kspace.dtype)
            num_slices = kspace.shape[0]
            num_coils = kspace.shape[1]
            for slice_idx in range( num_slices ):
                gt_ksp = kspace[slice_idx]
                s_maps_ind = bart(1, 'ecalib -m1 -W -c0', gt_ksp.transpose((1, 2, 0))[None,...]).transpose( (3, 1, 2, 0)).squeeze()
                s_maps[ slice_idx ] = s_maps_ind

            h5 = h5py.File( output_name, 'w' )
            h5.create_dataset( 's_maps', data = s_maps )
            h5.close()


if __name__ == '__main__':
    main(input_dir,output_dir)