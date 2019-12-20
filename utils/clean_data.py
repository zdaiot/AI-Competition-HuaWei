import os
import shutil
import json
import glob
import hashlib


def delete_orphaned(ipath):
    """ 删除当前路径下孤立的txt文件以及img文件

    Args:
        ipath: str，路径
    """
    filenames = os.listdir(ipath)
    txt_filenames = [file for file in os.listdir(ipath) if file.endswith('txt')]
    for txt_filename in txt_filenames:
        if txt_filename.replace('txt', 'jpg') not in filenames:
            print(txt_filename)
            os.remove(os.path.join(ipath, txt_filename))
    
    jpg_filenames = [file for file in os.listdir(ipath) if file.endswith('jpg')]
    for jpg_filename in jpg_filenames:
        if jpg_filename.replace('jpg', 'txt') not in filenames:
            print(jpg_filename)
            os.remove(os.path.join(ipath, jpg_filename))


def delete_acc_md5(ipath="data/huawei_data/official_google_bing"):
    """ 查询当前文件夹所有文件的md5值，删除md5值相同的图片和对应的txt
    Args:
        ipath: str，路径
    """
    def get_md5(filename):
        m = hashlib.md5()
        mfile = open(filename, "rb")
        m.update(mfile.read())
        mfile.close()
        md5_value = m.hexdigest()
        return md5_value

    filenames = [file for file in os.listdir(ipath) if file.endswith('jpg')]
    md5_dir = {}
    count = 0
    for filename in filenames:
        current_md5 = get_md5(os.path.join(ipath, filename))
        if current_md5 in md5_dir.keys():
            md5_dir[current_md5].append(filename)
            count += 1
            os.remove(os.path.join(ipath, filename))
        else:
            md5_dir[current_md5] = [filename]
    for key, values in md5_dir.items():
        if len(values) != 1:
            print(key + ":", values)

    # 因为上面只会删除图片，所以要再单独删除单个的txt文件
    delete_orphaned(ipath)


def clean_data(dataset_root, label_id):
    """ 如果txt名称和txt类标对应的名称不一致则删除该数据

    Args:
        dataset_root: str, 要清洗的数据集路径
        label_id: dict, {名称：label}
    """
    count = 0
    txt_paths = glob.glob(os.path.join(dataset_root, '*.txt'))
    # 处理每一个txt
    for txt_path in txt_paths:
        # 如果不是官方数据集
        if not txt_path.split('/')[-1].startswith('img'):
            # 打开文件
            with open(txt_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    txt_name = line.split(', ')[0]
                    label_index = int(line.split(', ')[1])
            label_name = label_id[str(label_index)]
            # 如果txt名称和txt类标对应的名称不一致
            if txt_name.split('_')[0] not in label_name:
                if txt_name.split('_')[0] == '浆水鱼鱼' and label_name == '美食/凉鱼':
                    pass
                else:
                    count += 1
                    print(txt_name, label_name)
                    os.remove(txt_path)
                    os.remove(txt_path.replace('.txt', '.jpg'))

    print('Deal {} image'.format(count))


def choose_data(choose_dict):
    """ 用于从数据集中筛选只出现在choose_dict中的样本，并复制到save_path中

    Args:
        choose_dict: dict, {名称：label}
    """
    choose_classes = choose_dict.values()
    choose_classes = [x.split('/')[-1] for x in choose_classes]

    origin_path = './psudeo_image'
    save_path = './choose_image'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for name in os.listdir(origin_path):
        if name.split('_')[0] in choose_classes:
            print('copy file {}'.format(name))
            shutil.copy(os.path.join(origin_path, name), save_path)


def split_official_pseudo(origin_path, save_path):
    """ 将包含官方数据和自己数据的文件夹中将自己数据复制到save_path

    Args: 
        origin_path: str, 包含官方数据和自己数据的路径
        save_path: str, 自己的数据放置的位置
    """
    # shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name in os.listdir(origin_path):
        if not name.startswith('img_'):
            shutil.copy(os.path.join(origin_path, name), save_path)

    
def delete_remote_data():
    """ 
    用于对比远程数据和本地数据的不同，注意需要使用`ls > filenames.txt`指令将本地文件夹的所有文件名称填充到`filenames.txt`文件中，并上传到`s3://ai-competition-zdaiot/data/`文件夹中

    see https://github.com/huaweicloud/ModelArts-Lab/blob/master/docs/moxing_api_doc/MoXing_API_File.md
    """
    import os
    try:
        import moxing as mox
    except:
        print('not use moxing')
        
    # mox.file.copy_parallel('s3://ai-competition-zdaiot/data/delete/', 's3://ai-competition-zdaiot/data/official_google_bing')

    local_data_root = '/cache/'
    mox.file.copy('s3://ai-competition-zdaiot/data/filenames.txt', local_data_root+'filenames.txt')

    with open(local_data_root+'filenames.txt') as f:
        data_list = f.readlines()
        data_list = [x.strip() for x in data_list]
        # print(data_list)

    remote_list = mox.file.list_directory('s3://ai-competition-zdaiot/data/official_google_bing', recursive=False)
    for x in remote_list:
        if x not in data_list:
            mox.file.remove('s3://ai-competition-zdaiot/data/official_google_bing/'+x, recursive=False)
            print(x)

    for x in data_list:
        if x not in remote_list:
            print(x)

if __name__ == '__main__':
    with open("data/huawei_data/label_id_name.json", 'r', encoding='utf-8') as json_file:
        label_id = json.load(json_file)
    # clean_data('data/huawei_data/combine', label_id)

    split_official_pseudo('data/huawei_data/combine', 'data/huawei_data/psudeo_image')