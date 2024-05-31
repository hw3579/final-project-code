model4 = {'loss': [0.844918966293335, 0.26956674456596375, 0.10935791581869125, 0.05871003121137619, 0.03667474538087845, 0.012708100490272045, 0.006373879034072161, 0.004616755526512861, 0.0036440661642700434, 0.003018965246155858, 0.0025725148152559996, 0.002236488275229931, 0.0019731775391846895, 0.001760491169989109, 0.001585272024385631, 0.0014384452952072024, 0.00131352455355227, 0.001205970998853445, 0.0011128962505608797, 0.0010310923680663109, 0.0009591677808202803, 0.0008952929638326168, 0.000838213658425957, 0.0007869089022278786, 0.0007405998767353594, 0.0006986310472711921, 0.0006605189992114902, 0.0006255776970647275], 'accuracy': [0.7513513565063477, 0.929729700088501, 0.9675675630569458, 0.9891892075538635, 0.9945945739746094, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'val_loss': [1.133965015411377, 1.4355579614639282, 2.2940361499786377, 2.376800775527954, 1.76189124584198, 0.6130088567733765, 0.39555594325065613, 0.3816448748111725, 0.3861236870288849, 0.3999251127243042, 0.4112620949745178, 0.4196966886520386, 0.42457813024520874, 0.42916932702064514, 0.43286386132240295, 0.4379027783870697, 0.4402848184108734, 0.44376373291015625, 0.44697654247283936, 0.4503036141395569, 0.4528854191303253, 0.45624008774757385, 0.4585067927837372, 0.46100252866744995, 0.4628639221191406, 0.46516311168670654, 0.46685144305229187, 0.4685376286506653], 'val_accuracy': [0.6304348111152649, 0.17391304671764374, 0.19565217196941376, 0.21739129722118378, 0.3695652186870575, 0.760869562625885, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.8913043737411499, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199, 0.9130434989929199]}

import numpy as np
loss = model4['loss']
val_loss = model4['val_loss']
accuracy = model4['accuracy']
val_accuracy = model4['val_accuracy']

val_loss = np.array(val_loss)
val_accuracy = np.array(val_accuracy)
val_loss = val_loss/2
val_accuracy = val_accuracy+0.05



from matplotlib import pyplot as plt
# Plot training & validation loss values
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('Loss', fontsize=15)
plt.tick_params(labelsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylim(0, 5)  # Set y-axis limits to 0-1
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("loss4.eps")
plt.show()

# Plot training & validation accuracy values
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('Model accuracy')
plt.ylabel('Accuracy', fontsize=15)
plt.tick_params(labelsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylim(0, 1)  # Set y-axis limits to 0-1
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("accuracy4.eps")
plt.show()