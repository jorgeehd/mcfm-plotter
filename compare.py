from utils import plot_axis_name, get_kfactor, arrange_data
from dataset import Dataset


import ROOT
import numpy as np
import os


class Compare: 

    def __init__(self, datasets: list, name): 
        self.datasets = datasets 
        self.var = datasets[0].var
        self.kfactor = None 
        self.name = name

        for i, dataset in enumerate(self.datasets): 
            if not isinstance(dataset, Dataset):
                raise ValueError("All elements in datasets must be of type Dataset.")
            if dataset.var != self.var: 
                raise ValueError(f"Dataset {i} has variable {dataset.var}. The variable for this compare object is {self.var}. All datasets must be for the same variable.")
        


    def create_kfactor(self, num_dataset , denom_dataset , new_y_axis_name:str , name:str = None , output_dir="k_factors" ,save = True):
        """
        Creates a k-factor histogram from the ratio of two datasets. 
        """
        os.makedirs(output_dir, exist_ok=True)
        if num_dataset not in self.datasets or denom_dataset not in self.datasets:
            raise ValueError("Both datasets must be part of the Compare object.")
        if num_dataset.var != self.var or denom_dataset.var != self.var:
            raise ValueError("Both datasets must be for the same variable.")
        
        
        # kfactor name is f"kfactor_{num_dataset.name}_over_{denom_dataset.name}_{var}" 


        if num_dataset.hist is None or denom_dataset.hist is None:
            num_dataset.make_hist()
            denom_dataset.make_hist()

        k_factorHist = num_dataset.hist.Clone("tmp") 
        k_factorHist.GetYaxis().SetTitle(new_y_axis_name)

        if num_dataset.hist or denom_dataset.hist is not None: 
            k_factorHist.Divide(num_dataset.hist ,denom_dataset.hist) 
            
        else: 
            k_factorHist = get_kfactor(num_dataset.data, denom_dataset.data, self.var)

        if name is None: 
            k_factorData = Dataset.from_rootHist(name = f"kfactor_{num_dataset.name}_over_{denom_dataset.name}_{self.var}" , var = self.var , hist=k_factorHist )
        else: 
            k_factorData = Dataset.from_rootHist(name = name, var= self.var , hist=k_factorHist)
            
        # save kfactor histogram
        if save: 
            k_factorData.save_as_TH1F(filename = f"kfactor_{self.var}" , output_dir = f"{output_dir}/{k_factorData.name}")
        
        self.kfactor = k_factorData 
        return None

    def scale_histogram(self, input_dataset , name = None): 
        """
        Scales all datasets using the k-factor histogram.
        """
        if not isinstance(input_dataset, Dataset):
            raise ValueError("Argument for input dataset has to be a dataset object.")
        if input_dataset not in self.datasets: 
            raise ValueError("Input dataset not in the compare object datasets. Please add the desired input hist to the compare object first.")
        if self.kfactor is None:
            raise ValueError("K-factor histogram not created. Call create_kfactor() first.")
        
        
        scaled_dataset = input_dataset.scale_by(name = name, factor = self.kfactor.hist)

        self.add_dataset(scaled_dataset)
            
        
        return None
    

    def add_dataset(self, dataset):
        """Add a new Dataset to the comparison, enforcing consistency."""
        if not isinstance(dataset, Dataset):
            raise TypeError("Only objects of type Dataset can be added.")
        if dataset.var != self.var:
            raise ValueError(
                f"Dataset variable mismatch: {dataset.var} (expected {self.var})"
            )
        self.datasets.append(dataset)
        print(f"Added dataset '{dataset.name}' to comparison ({len(self.datasets)} total).")


    
    def plot(self, labels=None, colors=None, logy=False, output_dir= f"plots" , sel:list[bool] = None):
        """
        Plots the cross section, for a given variable, for two datasets overlayed. 
        
        Labels allows for customizing legend labels.  
        Colors allows for customizing plot colors
        logy allows to user to get the plots in log scale for y. Default is false for no log scales
        output_dir allows for customizing the end directory for the plots.
        sel allows for selecting which of the datasets in the compare object to plot. It is a list of boolean values. 
        """

        ROOT.setTDRStyle()

        if labels is None:
            labels = [f"Dataset {i+1}" for i in range(len(self.datasets))]
        if colors is None:
            colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta]

        #canvas 
        c = ROOT.TCanvas("c", "Comparison", 800, 600)
        c.SetGrid()

#        stack  = ROOT.THStack("my_stack", self.name)

        axNames = plot_axis_name(self.var).split(";")

        if sel is None: 
            for i, dataset in enumerate(self.datasets):
                if dataset.hist is None: 
                    dataset.make_hist()
                dataset.hist.SetLineColor(colors[i%len(colors)])
                dataset.hist.SetLineWidth(2)
                dataset.hist.SetMarkerSize(0.5)
                dataset.hist.SetTitle(f";{axNames[1]};{axNames[2]}")
                if i>0:
                    dataset.hist.Draw("SAME") #stack.Add( ) 
                else:
                    dataset.hist.Draw()


                # legend
            leg = ROOT.TLegend(0.7, 0.8, 0.9, 0.95)
            leg.SetFillStyle(0) # transparent background
            leg.SetBorderSize(0) # no border
            leg.SetTextFont(62)

            for d, label in zip(self.datasets, labels):
                leg.AddEntry(d.hist, label, "l")
            leg.Draw()                

        else: 
            #have to do this for masking because lists dont have masking only pandas/numpy arrays. 
            tmp_datasets = [d for d, keep in zip(self.datasets, sel) if keep] #lol
            for i, dataset in enumerate(tmp_datasets):
                if dataset.hist is None: 
                    dataset.make_hist()
                dataset.hist.SetLineColor(colors[i%len(colors)])
                dataset.hist.SetLineWidth(2)
                dataset.hist.SetMarkerSize(0.5)
                dataset.hist.SetTitle(f";{axNames[1]};{axNames[2]}")
                if i>0:
                    dataset.hist.Draw("SAME") #stack.Add( ) 
                else:
                    dataset.hist.Draw()   

            # legend
            leg = ROOT.TLegend(0.7, 0.8, 0.9, 0.95)
            leg.SetFillStyle(0) # transparent background
            leg.SetBorderSize(0) # no border
            leg.SetTextFont(62)

            for d, label in zip(tmp_datasets, labels):
                leg.AddEntry(d.hist, label, "l")
            leg.Draw()
                     


        # saveplo
        os.makedirs(f"{output_dir}/{self.name}", exist_ok=True)
    
        c.SaveAs(f"{output_dir}/{self.name}/{self.var}.pdf")
        c.SaveAs(f"{output_dir}/{self.name}/{self.var}.png")
        if logy: 
            c.SetLogy(1)
            c.Update()
            c.SaveAs(f"{output_dir}/{self.name}/{self.var}_log.pdf")
            c.SaveAs(f"{output_dir}/{self.name}/{self.var}_log.png")



        return None            

            
    