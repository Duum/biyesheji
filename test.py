from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
with open("test_content.bin","rb") as f:
   content=np.load(f)
X_test=content.astype(np.float32)
with open("test_result.bin","rb") as f:
    result=np.load(f)
Y_test=np.expand_dims(result, axis=1).astype(np.float32)
model = model_from_json(open('model1.json').read())
model.load_weights("hehe1.h5")
hehe=model.predict(X_test)
result=[]
for item in hehe:
    if item<0.5:
        result.append(0)
    if item>0.5:
        result.append(1)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, hehe, pos_label=1)
auc_value=metrics.auc(fpr, tpr)
score=model.evaluate(X_test,Y_test)
plt.plot(fpr,tpr)
plt.show()
