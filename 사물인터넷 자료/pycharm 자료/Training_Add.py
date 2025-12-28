print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utils_Add import *
from tensorflow.keras.models import load_model
##############################################################################################
###### 확인 필수  (여기서 변경)

## Traning_Add 에서 확인할 것
# step1 path - 데이터 위치 (\\)
data_path = "C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\File\\AIoT"
# step7 load model - 기존 모델 파일 위치 (\\)
existing_model_path = "C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\venv\\suc_22.h5"
# step9 save model - 추가 학습한 모델 파일 이름
new_model_name = 'suc_final.h5'

## utils_Add 에서 확인할 것
# step1 importDataInfo - 학습 데이터 폴더 index [시작 번호, 끝 번호]
data_index = [11, 14]

##############################################################################################

#### STEP 1 - INITIALIZE DATA
path = data_path
data = importDataInfo(path, data_index)

#### STEP 2 - VISUALIZE AND BALANCE DATA
data = balanceData(data, display=True)

#### STEP 3 - PREPARE FOR PROCESSING
imagesPath, y_set= loadData(path,data)

### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, y_set,
                                               test_size=0.2, random_state=10)

print("y_train : ", yTrain)
#### STEP 5 - AUGMENT DATA

#### STEP 6 - PREPROCESS

#### STEP 7 - CREATE MODEL
# model = load_model('/폴더위치/모델이름.h5')
model = load_model(existing_model_path)

#### STEP 8 - TRAINNING
# dataGen(imagesPath, steeringList, batchSize, trainFlag):
history = model.fit(dataGen(xTrain, yTrain, 300, 1),
                                  steps_per_epoch=300,
                                  epochs=5,
                                  validation_data=dataGen(xVal, yVal, 300, 0),
                                  validation_steps=300)

#### STEP 9 - SAVE THE MODEL
model.save(new_model_name)
print('Model Saved')

#### STEP 10 - PLOT THE RESULTS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.show()