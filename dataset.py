from utils import arrange_data, histogram_filler 

import ROOT
import os 

class Dataset: 
    def __init__(self,  var: str , name: str , directory: str = None, data = None, hist = None):
        self.name = name 
#        self.directory = directory
        self.var = var 
        self.data = data 
        self.hist = hist 

    @classmethod
    def from_rootHist(cls, hist:ROOT.TH1F, var:str , name: str ): 
        return  cls(name = name, var = var, hist = hist) 
    
    @classmethod
    def from_directory(cls, directory: str, var: str, name: str):
        data = arrange_data(directory, var)
        hist = histogram_filler(data, var , name)
        return cls( var = var, name = name , directory= directory, data = data , hist = hist)
    
    @classmethod
    def  from_rootfile(cls, filepath: str, histname: str, var: str, name: str): 
        file = ROOT.TFile.Open(filepath)
        if not file or file.IsZombie():
            raise IOError(f"Cannot open ROOT file: {filepath}")
        hist = file.Get(histname)
        if not hist:
            raise KeyError(f"Histogram '{histname}' not found in {filepath}")
        hist = hist.Clone(f"{name}")
        hist.SetDirectory(0)  
        file.Close()
        return cls(name=name or histname, var=var, hist=hist)

    def load_data(self):
        self.data = arrange_data(self.directory, self.var)
        return self.data
    
    def make_hist(self): 
        if self.hist is not None:
            print("Histogram already exists, returning existing histogram.")
            return self.hist
        if self.data is None: 
            self.load_data()
        self.hist = histogram_filler(self.data, self.var)
        return self.hist

    def scale_by(self, name: str, factor: float):   

        if self.hist is None: 
            self.make_hist()

        if name is None: 
            scaled_dataset = Dataset(name = f"scaled_{self.name}" , var = self.var )
        else:
            scaled_dataset=Dataset(name = f"{name}" , var = self.var )

        scaled_dataset.hist = self.hist.Clone()
            
        #no functionality for scalars like doubles and ints yea i know bad name for function for now  
        if type(factor) == Dataset: 
            scaled_dataset.hist.Multiply(factor.hist)

        elif type(factor) == ROOT.TH1F: 
            scaled_dataset.hist.Multiply(factor)
       
        #print(f"Var: {scaled_dataset.var}")

        return scaled_dataset


    def plot(self, color=ROOT.kRed, logy=False):
        if self.hist is None:
            self.make_hist()
        c = ROOT.TCanvas(f"c_{self.name}", f"{self.var}", 800, 600)
        self.hist.SetLineColor(color)
        self.hist.Draw()
        if logy:
            c.SetLogy(1)
        return c
    
    def save_plot(self, filename: str, color=ROOT.kBlue, logy=False):
        c = self.plot(color=color, logy=logy)
        c.SaveAs(filename)
        return filename
    
    def save_as_TH1F(self, filename: str, output_dir="rootFiles" ):

        if self.hist is None:
            self.make_hist()
        # saveplot a
        os.makedirs(f"{output_dir}", exist_ok=True)

        f = ROOT.TFile(f"{output_dir}/{filename}.root", "RECREATE")
        self.hist.Write(self.name)
        f.Close()
        return f"Histogram file saved to {output_dir}/f{filename}.root"

