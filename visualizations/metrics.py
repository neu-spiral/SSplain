import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    """Top-k accuracy over the specified values of k (from the reference repo)."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def Deletion(model, input_img, target_label, mask, explainer_name, step = 25):  
    '''
    Calculate balanced accuracy for each deletion step s 
    
    Inputs: 
        model: Trained model
        input_img: Input iamges
        target_label: Target labels
        mask: Attribution scores
        explainer_name: Name of the explainer method
        step: Number of deletion steps
    Outputs:
        np_scores: Balanced accuracy for each deletion step s
        np_norm: Normalized norm (kappa_s from the paper)
        
    '''  
    explanations = mask.detach().clone()
    if explanations.shape[1]==3:
        nb_features = np.prod(explanations.shape[1:])
        explanations_flatten =  torch.reshape(explanations,(explanations.shape[0],nb_features))
        
        nb_features = np.prod(input_img.shape[1:])
        input_img_flatten = torch.reshape(input_img,(input_img.shape[0], nb_features)).type(dtype=torch.float32)
        most_important_features = torch.argsort(explanations_flatten, descending = True )    
        steps = np.linspace(0, most_important_features.shape[1], step + 1, dtype=np.int32)
    else:    
        nb_features = np.prod(explanations.shape[2:])
        explanations_flatten =  torch.reshape(explanations,(explanations.shape[0], explanations.shape[1],nb_features))
        
        nb_features = np.prod(input_img.shape[2:])
        input_img_flatten = torch.reshape(input_img,(input_img.shape[0],input_img.shape[1], nb_features)).type(dtype=torch.float32)
        most_important_features = torch.argsort(explanations_flatten, descending = True )
        steps = np.linspace(0, most_important_features.shape[2], step + 1, dtype=np.int32)
    baselines = torch.zeros_like(input_img,dtype=torch.float32)
    baselines_flatten = torch.reshape(baselines, (input_img_flatten.shape))
    
    
    start = input_img_flatten
    end = baselines_flatten   
    
    scores_dict = {}
    norm_dict = {}
    norm_full = torch.norm(input_img.float(), p=1, dim = [1,2,3]).mean()
    for step_size in steps:
        if explanations.shape[1]==3: 
            ids_to_flip = most_important_features[:, :step_size]
        else:
            ids_to_flip = most_important_features[:,:, :step_size]
        batch_inputs = start.detach().clone()
        for i, ids in enumerate(ids_to_flip):
            if explanations.shape[1]==3:  
                batch_inputs[i, ids] = end[i, ids]
            else:
                batch_inputs[i,0, ids[0]] = end[i,0, ids[0]]

        batch_inputs = torch.reshape(batch_inputs,(input_img.shape)) 
        with torch.no_grad():
            output = model(batch_inputs)
        acc1, acc5 = accuracy(output, target_label, topk=(1, 5))
        scores_dict[step_size] = acc1.cpu()
        norm_dict[step_size] = torch.norm(batch_inputs.float(), p=1, dim = [1,2,3]).cpu().mean()/norm_full.cpu()
        

    np_scores = np.array(list(scores_dict.values()))
    np_norm = np.array(list(norm_dict.values()))
    return np_scores, np_norm


def Insertion(model, input_img, target_label, mask, explainer_name, step = 25): 
    '''
    Calculate balanced accuracy for each insertion step s 
    
    Inputs: 
        model: Trained model
        input_img: Input iamges
        target_label: Target labels
        mask: Attribution scores
        explainer_name: Name of the explainer method
        step: Number of insertion steps
    Outputs:
        np_scores: Balanced accuracy for each insertion step s
        np_norm: Normalized norm (kappa_s from the paper)
        
    '''      
    explanations = mask.detach().clone()
    if explanations.shape[1]==3:
        nb_features = np.prod(explanations.shape[1:])
        explanations_flatten =  torch.reshape(explanations,(explanations.shape[0],nb_features))
        
        nb_features = np.prod(input_img.shape[1:])
        input_img_flatten = torch.reshape(input_img,(input_img.shape[0], nb_features)).type(dtype=torch.float32)
        most_important_features = torch.argsort(explanations_flatten, descending = True )    
        steps = np.linspace(0, most_important_features.shape[1], step + 1, dtype=np.int32)
    else:    
        nb_features = np.prod(explanations.shape[2:])
        explanations_flatten =  torch.reshape(explanations,(explanations.shape[0], explanations.shape[1],nb_features))
        
        nb_features = np.prod(input_img.shape[2:])
        input_img_flatten = torch.reshape(input_img,(input_img.shape[0],input_img.shape[1], nb_features)).type(dtype=torch.float32)
        most_important_features = torch.argsort(explanations_flatten, descending = True )
        steps = np.linspace(0, most_important_features.shape[2], step + 1, dtype=np.int32)
   
    baselines = torch.zeros_like(input_img,dtype=torch.float32) 
    baselines_flatten = torch.reshape(baselines, (input_img_flatten.shape))
            
    start = baselines_flatten 
    end = input_img_flatten  
    
    scores_dict = {}
    norm_dict = {}
    norm_full = torch.norm(input_img.float(), p=1, dim = [1,2,3]).mean()
    for step_size in steps:
        if explanations.shape[1]==3: 
            ids_to_flip = most_important_features[:, :step_size]
        else:
            ids_to_flip = most_important_features[:,:, :step_size]
        batch_inputs = start.detach().clone()
        
        for i, ids in enumerate(ids_to_flip):
            if explanations.shape[1]==3:  
                batch_inputs[i, ids] = end[i, ids]

            else:
                batch_inputs[i,0, ids[0]] = end[i,0, ids[0]]
      
        batch_inputs = torch.reshape(batch_inputs,(input_img.shape))  
        with torch.no_grad():
            output = model(batch_inputs)
        acc1, acc5 = accuracy(output, target_label, topk=(1, 5))
        scores_dict[step_size] = acc1.cpu()
        norm_dict[step_size] = torch.norm(batch_inputs.float(), p=1, dim = [1,2,3]).cpu().mean()/norm_full.cpu()
    np_scores = np.array(list(scores_dict.values()))
    np_norm = np.array(list(norm_dict.values()))
    return np_scores, np_norm
