# to annotate data and preparing annotation files

from __future__ import print_function, division
import os
import sys
import subprocess
import zipfile

if __name__=="__main__":
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  #unzip all files..  
  """for class_name in os.listdir(dir_path):
    print(class_name)
    for subject_name in os.listdir(os.path.join(dir_path, class_name)):
        zip_ref = zipfile.ZipFile(os.path.join(dir_path, class_name, subject_name), 'r')
        zip_ref.extractall(os.path.join(dst_dir_path, class_name))
        zip_ref.close()
  """
  #tr = open('/home/sonu/Downloads/GRIT/HRI_GestureDataset_Tsironi/trainlist.txt', 'a')
  #vl = open('/home/sonu/Downloads/GRIT/HRI_GestureDataset_Tsironi/vallist.txt', 'a')
  #te = open('/home/sonu/Downloads/GRIT/HRI_GestureDataset_Tsironi/testlist.txt', 'a')
  for class_name in os.listdir(dir_path):
      #print(class_name)
   # class_process(dir_path, dst_dir_path, class_name)
      if not os.path.isdir(os.path.join(dir_path, class_name)):
          continue

      file_list = [[] for i in range(7)]
      for file_name in os.listdir(os.path.join(dir_path, class_name)):
          #print(file_name)
          if '.zip' in file_name:
              subprocess.call('rm -r \"{}\"'.format(os.path.join(dir_path, class_name, file_name)), shell=True)
          else:
              print(file_name)
          # used for renaming
          #os.rename(os.path.join(dir_path, class_name, file_name), os.path.join(dir_path, class_name, class_name + '_' + file_name))

          #jepg files renaming
          list_name = []
          for jpeg_file in os.listdir(os.path.join(dir_path, class_name, file_name)):
              #os.rename(os.path.join(dir_path, class_name, file_name, jpeg_file), os.path.join(dir_path, class_name, file_name, 'image{:2d}'.format(i)))
              list_name.append(jpeg_file)
          list_name.sort()
          i = 1
          print(len(list_name))
          for j in range(0, len(list_name)):
              file = list_name[j]
              os.rename(os.path.join(dir_path, class_name, file_name, file),
                      os.path.join(dir_path, class_name, file_name, 'image'+'_{:2d}.jpg'.format(j)))

          #split = file_name.split('_')
          #file_list[int(split[-2])].append(file_name)

      #print(file_list)
      # prepare 60:20:20 dataset for training, validation and testing
      """for i in range(1, 7):
          for j in range(len(file_list[i])):
              if j < 6:
                tr.write('{}/{}\n'.format(class_name, file_list[i][j]))
              else:
                  if j % 2 == 0:
                      vl.write('{}/{}\n'.format(class_name, file_list[i][j]))
                  else:
                      te.write('{}/{}\n'.format(class_name, file_list[i][j]))
      file_list = []
#  tr.close()
# te.close()
# vl.close()
"""