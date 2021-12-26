#%%
import os.path
import time
modified_time = os.path.getmtime(r'./saved_model')
print(modified_time)
convert_time = time.ctime(modified_time)
print(convert_time)
# %%
folder_name = ''
last_modified = 0.00
for dir in os.listdir('./saved_model'):
    if(dir != 'my_model'):
        if(float(dir) > float(last_modified)):
            last_modified = dir

print(str(last_modified))

# %%
