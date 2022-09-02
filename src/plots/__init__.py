loss = [0.425, 0.414, 0.445, 0.428, 0.424, 0.410]
acc = [0.871, 0.891, 0.884, 0.888, 0.885, 0.893]
tim = [102.116, 101.616, 101.724, 103.730, 101.895, 102.863]
epoch=[245, 246,247,248,249,250]
for j in range(6):
    print("-------- Epoch {0} Loss : {1} Accuracy : {2} Time : {3} --------".format(epoch[j],loss[j],acc[j],tim[j]))
print("Finished Training")