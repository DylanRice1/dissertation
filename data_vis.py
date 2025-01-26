from imports import *

experiment_name = 'Transfer Learning'

data_dir = './UsableData/'
train_path = os.path.join(data_dir,'train')
test_path = os.path.join(data_dir,'test')
val_path = os.path.join(data_dir,'val')

fig, axes = plt.subplots(2, 2, figsize = (5,5)) 
axes = axes.ravel() 

for i in np.arange(0, 4): 

    category = random.choice(os.listdir(train_path)) 
    class_dir = os.path.join(train_path, category)

    image = random.choice(os.listdir(class_dir)) 
 
    img = plt.imread(os.path.join(class_dir,image))
    axes[i].imshow( img )
    axes[i].set_title(category) 
    axes[i].axis('off')

plt.show()


# Looking at distribution of data between classes
total = 0
for category in os.listdir(train_path):
    count= 0
    for image in os.listdir(train_path + "/" + category):
        count += 1
        total +=1
    print(str(category).title() + ": " + str(count))  
print(f"\nTotal number of train images: {total}")

# class names
class_names = sorted(os.listdir(train_path))
print(class_names)

class_dis = [len(os.listdir(train_path + f"/{name}")) for name in class_names]
print(class_dis)

# Visualising the distribution
DF = pd.DataFrame(columns=['Class names','Count'])
DF['Class names']=class_names
DF['Count']=class_dis
plt.figure(figsize=(7,4))
ax=sns.barplot(x='Class names', y='Count', data=DF)
ax.bar_label(ax.containers[0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


# # To fix the imbalance of data samples, randomly choosen elements of the larger classes will be removed
# for category in os.listdir(train_path):
    
#     count = 0
#     delete = 0
#     for image in os.listdir(train_path + '/' + category):
#         count += 1
#         while count > 1000:
#             random_image = random.choice(os.listdir(train_path + '/' + category))
#             delete_image = train_path + '/' + category + '/' + random_image
#             os.remove(delete_image)
#             delete +=1
#             count -= 1
#     print(f'Deleted {delete} in {category}')


class_dis = [len(os.listdir(train_path + f"/{name}")) for name in class_names]

DF = pd.DataFrame(columns=['class','count'])
DF['class']=class_names
DF['count']=class_dis
plt.figure(figsize=(7,4))
ax=sns.barplot(x='class', y='count', data=DF)
ax.bar_label(ax.containers[0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


IMG_SIZE = (400, 400)
SEED = 53
BATCH_SIZE = 32

writer = SummaryWriter()

# Building training data with appropriate tf parameters
train_data = tf.keras.utils.image_dataset_from_directory(train_path,
                                                         image_size = IMG_SIZE,
                                                         label_mode = 'categorical',
                                                         batch_size = BATCH_SIZE,
                                                         shuffle = True,
                                                         seed = SEED)

# Building testing data with appropriate tf parameters
test_data = tf.keras.utils.image_dataset_from_directory(test_path,
                                                        image_size=IMG_SIZE,
                                                        label_mode='categorical',
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)

# Building val data with appropriate tf parameters
validation_data = tf.keras.utils.image_dataset_from_directory(val_path,
                                                              image_size=IMG_SIZE,
                                                              label_mode='categorical',
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              seed=SEED)

# for file in validation_data:
#     print(file)


train_log_dir = os.path.join("logs", experiment_name, "train")
test_log_dir = os.path.join("logs", experiment_name, "test")

os.makedirs(train_log_dir, exist_ok=True)
os.makedirs(test_log_dir, exist_ok=True)

train_writer = SummaryWriter(train_log_dir)
test_writer = SummaryWriter(test_log_dir)

# Defining the base model (ResNet50)
base_model = tf.keras.applications.resnet50.ResNet50(include_top = False,
                                                     weights = 'imagenet',
                                                     input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

