import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utilities import bootstraps

def predict(modelfile, net, df, data_root, transformer, binary, dense, targets, gradcam):
    print("Predict:", modelfile, df.shape)

    dataset = Dataset(df, size = 420, data_root= data_root, transform = transformer, binary = binary, 
                      dense = dense, targets=targets)

    data_loader = torch_data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory = True)
    
    if binary:
        classes = 2
    else:
        classes = 4
   
    model = Model(net = net, classes = classes )
    
    model.to(device)
    
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    if gradcam:
        model = medcam.inject(model, output_dir='../attention_maps/gradcampp/test/best', backend='gcampp',
                              data_shape=[150,420,144],label='best', save_maps=True)

    #print(model)
    
    
    
    y_prob_0 = []
    y_prob_1 = []
    y_prob_2 = []
    y_prob_3 = []
    y_pred = []
    ids = []
    labels = []
    
    if gradcam:
        for e, batch in enumerate(data_loader,1):
            print(f"{e}/{len(data_loader)}", end="\r")
            _ = model(batch["X"].to(device))
            
    else:
        for e, batch in enumerate(data_loader,1):
            print(f"{e}/{len(data_loader)}", end="\r")
            with torch.no_grad():
                
                tmp_prob = F.softmax(model(batch["X"].to(device)),1).cpu().numpy()
                _,tmp_pred = torch.max(model(batch["X"].to(device)),1)
                pred  = tmp_pred.cpu().numpy()

               
                y_pred.extend(pred)

                ids.extend(batch["id"])
                labels.extend(batch["y"].numpy().tolist())
                
                
                if binary:
                    y_prob_0.extend(tmp_prob[:,0])
                    y_prob_1.extend(tmp_prob[:,1])
                else:
                    y_prob_0.extend(tmp_prob[:,0])
                    y_prob_1.extend(tmp_prob[:,1])
                    y_prob_2.extend(tmp_prob[:,2])
                    y_prob_3.extend(tmp_prob[:,3])
                    
        if binary:
            preddf = pd.DataFrame({"patient_ID": ids ,"Label": labels, "Prediction": y_pred, \
                               "Probility_0": y_prob_0,"Probility_1": y_prob_1
                              })
            
        else:

            preddf = pd.DataFrame({"patient_ID": ids ,"Label": labels, "Prediction": y_pred, \
                                   "Probility_0": y_prob_0,"Probility_1": y_prob_1,
                                   "Probility_2": y_prob_2,"Probility_3": y_prob_3
                              })
        #preddf = preddf.set_index("patient_ID")
        return preddf
