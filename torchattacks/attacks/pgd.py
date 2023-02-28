import torch
import torch.nn as nn
import pdb

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels,mode = None,mode2=None,schedule = 0,verbose = False):
        r"""
        Overridden.
        """

        if mode in [102,103,104,105]:
            #pdb.set_trace()
            run_backwards = False
            if mode == 105:
                run_backwards = True
                mode = 103
            img = images["observed_data"].detach().to(self.device)
            labels = images["data_to_predict"].detach().to(self.device)
            obs_mask = images["observed_mask"].detach().to(self.device)
            init = images['observed_init'].to(self.device)

            if mode in [104]:
                labels = images["labels"].detach().to(self.device)


            pred_mask = images["mask_predicted_data"].detach().to(self.device)
            idx_mask = torch.where(pred_mask == 1)

            adv_images = img.clone()
            adv_init = init.clone()
            temp = 0
            boolean = mode in [103,104]
            if mode == 104 and labels == 1:
                temp = 100000
            if mode == 104 and labels == 0:
                temp = -10000

            criterion = nn.BCEWithLogitsLoss()
            if self.random_start:
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images[torch.where(obs_mask==0)] *= 0
                if boolean:
                    adv_init = adv_init + torch.empty_like(adv_init).uniform_(-self.eps, self.eps)

            adv_images[:,:,10] = torch.clamp(adv_images[:,:,10],0.0,1.0)
            adv_images[:,:,28] = torch.clamp(adv_images[:,:,28],0.0,1.0)
            
            for jj in range(self.steps):
                adv_images.requires_grad = True
                if boolean:
                    adv_init.requires_grad = True
                x = torch.cat((adv_images,obs_mask),dim=2).float().to(self.device)

                outputs, info = self.model(x,images["observed_tp"].to(self.device),images["tp_to_predict"].to(self.device),adv_init,target_mask = images['mask_predicted_data'],run_backwards=run_backwards)
                outputs = outputs.squeeze(0)

                if mode in [102,103]:
                    cost = (outputs[idx_mask] - labels[idx_mask]).abs().mean()
                    #print(cost)
                    if cost > temp:
                        final_adv_images = adv_images.clone()
                        temp = cost
                        if boolean:
                            final_init = adv_init.clone()
                else:
                    pred_labels = info["label_predictions"]
                    #print(pred_labels)
                    cost = criterion(pred_labels,labels)
                    if labels == 1 and  pred_labels < temp:
                        temp = pred_labels.clone()
                        final_adv_images = adv_images.clone()
                        if boolean:
                            final_init = adv_init.clone()
                    elif labels == 0 and  pred_labels > temp:
                        temp = pred_labels.clone()
                        final_adv_images = adv_images.clone()
                        if boolean:
                            final_init = adv_init.clone()

                    
                # Update adversarial images
                if boolean:
                    grad,grad1 = torch.autograd.grad(cost, (adv_images ,adv_init),
                                           retain_graph=False, create_graph=False)

                    adv_images = adv_images.detach() + self.alpha*grad.sign()
                    delta = torch.clamp(adv_images - img, min=-self.eps, max=self.eps)
                    adv_images = (img + delta).detach()
                    adv_images[torch.where(obs_mask==0)] = 0

                    adv_init = adv_init.detach() + self.alpha*grad1.sign()
                    delta = torch.clamp(adv_init - init, min=-self.eps, max=self.eps)
                    delta[:,1] = 0.
                    delta[:,3:] = 0.

                    adv_init = (init + delta).detach()
                else:
                    grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                    adv_images = adv_images.detach() + self.alpha*grad.sign()
                    delta = torch.clamp(adv_images - img, min=-self.eps, max=self.eps)
                    adv_images = (img + delta).detach()
                    adv_images[torch.where(obs_mask==0)] = 0
                adv_images[:,:,10] = torch.clamp(adv_images[:,:,10],0.0,1.0)
                adv_images[:,:,28] = torch.clamp(adv_images[:,:,28],0.0,1.0)



            x = torch.cat((adv_images,obs_mask),dim=2).float()
            outputs, info = self.model(x,images["observed_tp"],images["tp_to_predict"],adv_init,target_mask = images['mask_predicted_data'],run_backwards= run_backwards)
            outputs = outputs.squeeze(0)
            if mode in [102,103]:

                cost = (outputs[idx_mask] - labels[idx_mask]).abs().mean()
                if cost < temp:
                    if boolean:
                        return final_adv_images,final_init
                    return final_adv_images
            else:
                pred_labels = info["label_predictions"]
                if labels == 1 and  pred_labels >= temp:
                    if boolean:
                        return final_adv_images,final_init
                    return final_adv_images
                if labels == 0 and  pred_labels <= temp:
                    if boolean:
                        return final_adv_images,final_init
                    return final_adv_images

            if boolean:
                return adv_images,adv_init
            return adv_images

        if mode not in [100,101]:
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        if mode == None:
            loss = nn.CrossEntropyLoss()
        elif mode == 100: #mse increase autoregressive error
            meta_dict = images.copy()
            images = images["observed_data"].clone().detach().to(self.device)
            labels = labels.unsqueeze(0)
            loss = nn.MSELoss()
        elif mode == 101: #mse increase autoregressive error
            meta_dict = images.copy()
            images = images["observed_data"].clone().detach().to(self.device)
            labels = labels.unsqueeze(0).detach().to(self.device)
            loss = nn.L1Loss()
        else:
            loss = nn.MSELoss()
            labels = torch.zeros_like(labels)
        
        adv_images = images.clone().detach()
        
        
        if mode == 0:
            temp = labels.clone()
        elif mode ==1:
            temp = labels.clone() + 100
        elif mode in [100,101]:
            temp = 0
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            if mode not in  [101,100]:
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            else:
                adv_images[torch.where(meta_dict['observed_mask']==0)] *= 0
        for jj in range(self.steps):
            adv_images.requires_grad = True
            if mode in [100,101]:
                meta_dict["observed_data"] = adv_images
                outputs = self.model(meta_dict)
            else:
                outputs = self.model(adv_images)
            if verbose :
                print(self.model.net[5].err1)
            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            
            else:
                if mode2 == None or schedule == -1:
                #if True:
                    if mode in [None,100,101]:
                        #pdb.set_trace()
                        cost = loss(outputs, labels)
                    elif mode == 1:
                        cost = -1 * loss(self.model.net[5].err1[0], labels)
                    elif mode == 0:
                        cost = 1 * loss(self.model.net[5].err1[0], labels)

                    if mode2 == 1:
                        cost += -1 * loss(self.model.net[5].err1[1], labels)
                    elif mode2 == 0:
                        cost += 1 * loss(self.model.net[5].err1[1], labels)

                elif jj % schedule == 0:
                    if mode2 == 1:
                        cost = -1 * loss(self.model.net[5].err1[1], labels)
                    elif mode2 == 0:
                        cost = 1 * loss(self.model.net[5].err1[1], labels)
                else:
                    if mode == 1:
                        cost = -1 * loss(self.model.net[5].err1[0], labels)
                    elif mode == 0:
                        cost = 1 * loss(self.model.net[5].err1[0], labels)


            if mode == 1 and self.model.net[5].err1[0]< temp:
                temp = self.model.net[5].err1[0]
                final_adv_images =  adv_images.clone()
            elif mode == 0 and self.model.net[5].err1[0]> temp:
                temp = self.model.net[5].err1[0]
                final_adv_images =  adv_images.clone()
            elif mode in [100,101] and cost > temp:
                temp = cost
                final_adv_images =  adv_images.clone()

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)

            if mode not in [100,101]:
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            else:
                adv_images = (images + delta).detach()


        if mode == 1:
            outputs = self.model(adv_images)
            if self.model.net[5].err1[0] > temp:
                return final_adv_images
        elif mode == 0:
            outputs = self.model(adv_images)
            if self.model.net[5].err1[0] < temp:
                return final_adv_images
        elif mode in [100,101]:
            meta_dict["observed_data"] = adv_images
            outputs = self.model(meta_dict)
            cost = loss(outputs, labels)
            if cost < temp:
                return final_adv_images

        return adv_images
