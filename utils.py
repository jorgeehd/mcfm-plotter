import pandas 
import numpy as npe
import matplotlib.pyplot as plt
import os
import sys
import re
import argparse
import ROOT
import array
from ROOT import gStyle
from IPython.display import Image, display


ROOT.gROOT.LoadMacro("/Users/jorgehernandez/Documents/HEP_work/summer2024/macros/style/tdrstyle.C")
ROOT.gROOT.LoadMacro("/Users/jorgehernandez/Documents/HEP_work/summer2024/macros/style/CMS_lumi.C")
ROOT.setTDRStyle()
gStyle.SetOptStat(0)

def extract_data(filepath):
    """
    Extracts histogram data from a file and arranges it into a list of dictionaries with bin 
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 4:
                xmin = float(parts[0])
                xmax = float(parts[1])
                cross = float(parts[2])
                error = float(parts[3])
                data.append({'xmin': xmin, 'xmax': xmax, 'cross': cross, 'error': error})
    return data

def arrange_data(histogram_dir, var :str): 
    """
    Arranges histogram data from multiple files in a directory into 
    a single sorted list of dictionaries with bins, cross and error information.
    """
    hist = []
    files = [os.path.join(histogram_dir, f) for f in os.listdir(histogram_dir) if var in f]
#    print(files)
    if len(files) >1:
       files.sort(key=lambda x: int(re.search(fr'{var}_(\d+)', x).group(1))) #sort files 
    for file in files:
        hist.extend(extract_data(file))
    return hist 

def get_kfactor(data1, data2):
    """
    Calculates the bin-by-bin ratio of two datasets.
    data1, data2 : list of dicts
       each dict has keys: xmin, xmax, cross, error
    Returns a list of ratios.
    """
    if len(data1) != len(data2):
            raise ValueError("Datasets must have the same number of bins")
    
    k_factors = []
    
    for entry1, entry2 , k_entry in zip(data1, data2 , k_factors):

        if entry2["cross"] != 0:
            ratio = entry1["cross"] / entry2["cross"]
        else:
            ratio = 0
        
        error = npe.sqrt( (entry1["error"]/entry1["cross"])**2 + (entry2["error"]/entry2["cross"])**2 ) * ratio if entry1["cross"] !=0 and entry2["cross"] !=0 else 0
       
        k_factors.append({
            "xmin": entry1["xmin"],
            "xmax": entry1["xmax"],
            "cross": ratio,
            "error": error  # or compute propagated error if needed
        })
        #ratios.append(ratio)
        
    return k_factors

''' 
def scale_histogram(data, k_factors):
    """
    Scales a dataset by bin-by-bin k-factors.
    data, k_factors : list of dicts
       each dict has keys: xmin, xmax, cross, error
    Returns a new list of scaled data.
    """
    if len(data) != len(k_factors):
        raise ValueError("Data and k-factors must have the same number of bins")
    
    scaled_data = []
    
    for entry, k_entry in zip(data, k_factors):
        scaled_cross = entry["cross"] * k_entry["cross"]
        # Propagate error (assuming independent errors)
        scaled_error = npe.sqrt( (entry["error"]/entry["cross"])**2 + (k_entry["error"]/k_entry["cross"])**2 ) * scaled_cross if entry["cross"] !=0 and k_entry["cross"] !=0 else 0
        
        scaled_data.append({
            "xmin": entry["xmin"],
            "xmax": entry["xmax"],
            "cross": scaled_cross,
            "error": scaled_error
        })
        
    return scaled_data

'''

'''
def scale_histogram_wrapper(data1, data2, data_to_scale , var) -> ROOT.TH1F | dict:  
    """
    Wrapper function to calculate k-factors from data1 and data2, 
    then scale data_to_scale by these k-factors.
    """

    # Prepare data for manipulation. 
    data1 = arrange_data(data1 , var= var)
    data2 = arrange_data(data2 , var= var) 
    data_to_scale = arrange_data(data_to_scale , var= var)

    k_factors = get_kfactor(data1, data2)

    

    scaled_data = scale_histogram(data_to_scale, k_factors)
    return , scaled_data
'''

def plot_axis_name(var: str) -> str:
    """Return the appropriate axis labels based on the variable name."""
    mapping = {
        # diphoton mass
        # invariant mass
        "m12": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "m(12)": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "m34": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "m(34)": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "m(34).txt": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "m56": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "m(56)": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "mgg": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "mgaga": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]",
        "highMassRange": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]", 
        "highMass": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]", 
        "m34_highMass": ";m_{#gamma#gamma} [GeV]; d#sigma/dm [fb/GeV]", 

        # transverse momentum
        "ptgg": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt12": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(12)": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt34": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(34)": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(34).txt": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt3": ";pT_{#gamma,1} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(3)": ";pT_{#gamma,1} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt4": ";pT_{#gamma,2} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(4)": ";pT_{#gamma,2} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt5": ";pT_{#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(5)": ";pT_{#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt6": ";pT_{#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "pt(6)": ";pT_{#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "highPtRange": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",
        "highPtR": ";pT_{#gamma#gamma} [GeV]; d#sigma/dpT [fb/GeV]",

        # rapidity / angles
        "eta34": ";#eta_{#gamma#gamma}; d#sigma/d#eta [fb]",
        "eta(34)": ";#eta_{#gamma#gamma}; d#sigma/d#eta [fb]",
        "y34": ";y_{#gamma#gamma}; d#sigma/dy [fb]",
        "y(34)": ";y_{#gamma#gamma}; d#sigma/dy [fb]",
        "y3": ";y_{#gamma,1}; d#sigma/dy [fb]",
        "y(3)": ";y_{#gamma,1}; d#sigma/dy [fb]",
        "y4": ";y_{#gamma,2}; d#sigma/dy [fb]",
        "y(4)": ";y_{#gamma,2}; d#sigma/dy [fb]",
        "y5": ";y_{#gamma}; d#sigma/dy [fb]",
        "y(5)": ";y_{#gamma}; d#sigma/dy [fb]",

        "deltaphi": ";#Delta phi_{#gamma#gamma}; d#sigma/dphi [fb]",
        "DeltaR34": ";#Delta R_{#gamma#gamma}; d#sigma/dR [fb]",
        "Abs": ";|#eta_{#gamma#gamma}|; d#sigma/d|#eta| [fb]",
    }
    return mapping.get(var, var)

def histogram_filler(data1 , var , name = None ) -> ROOT.TH1F:
    """
    Fills a ROOT histogram from the provided data.
    data1 : list of dicts
       each dict has keys: xmin, xmax, cross, error
    Returns a ROOT.TH1F histogram.
    """
    # --- extract bin edges from first dataset ---
    bins = [entry["xmin"] for entry in data1]
    bins.append(data1[-1]["xmax"])
    nbins = len(bins) - 1
    # convert to C-style array for ROOT
    bin_edges = array.array("d", bins)
    
    # --- define histogram ---
    if name is not None:
        h = ROOT.TH1F(name, plot_axis_name(var), nbins, bin_edges)
    else:
        h = ROOT.TH1F("h", plot_axis_name(var), nbins, bin_edges)

    # fill histogram
    for i, entry in enumerate(data1):
        h.SetBinContent(i+1, entry["cross"])
        h.SetBinError(i+1, entry["error"])
    
    return h 

def plot1(directory , var: str , lg = None):
    """
    Plots histogram for one dataset.
    data1 : list of dicts
       each dict has keys: xmin, xmax, cross, error
    """
    data = arrange_data(directory , var )
    h = histogram_filler(data , var)
    h.SetLineColor(ROOT.kRed)
    h.SetLineWidth(2)
    c = ROOT.TCanvas("c", f"{var} Distribution", 800, 600)
    c.SetGrid()
    h.Draw()
    c.Draw()
    c.Update()
    # legend
    leg = ROOT.TLegend(0.6, 0.7, 0.85, 0.85)
    
    if lg is not None:
        leg.AddEntry(h, lg, "l")
    else:
        leg.AddEntry(h, directory, "l")
    leg.Draw()
    os.makedirs(f"{directory}_plots", exist_ok=True)
    c.SaveAs(f"{directory}_plots/diphoton_{var}.pdf")
    c.SaveAs(f"{directory}_plots/diphoton_{var}.png")
    return None



def plot(directories, var: str, labels=None, colors=None):
    """
    Overlay histograms from any number of datasets.
    
    Parameters
    ----------
    directories : list of str
        List of directories containing histogram files.
    var : str
        Variable to plot (e.g. "m34").
    labels : list of str, optional
        Legend labels (must match number of datasets).
    colors : list of int, optional
        ROOT colors (must match number of datasets).
    """
    if labels is None:
        labels = [f"dataset {i+1}" for i in range(len(directories))]
    if colors is None:
        # some distinct colors
        colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange, ROOT.kCyan]

    # build histograms
    print(f"len(directories): {len(directories)}" )
    histograms = []
    for i, d in enumerate(directories):
        print(i)
        dataset = arrange_data(d, var=var)
        h = histogram_filler(dataset , var = var , name = f"h{i}")
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(2)
        histograms.append(h)

    print(len(histograms))
    # axis titles
    axNames = plot_axis_name(var).split(";")

    # canvas
    c = ROOT.TCanvas("c", "Comparison", 800, 600)
    c.SetGrid()


    # draw histograms
    for i, h in enumerate(histograms):
        h.SetTitle(f";{axNames[1]};{axNames[2]}")
        opt = "" if i == 0 else "SAME"
        h.Draw(opt)

    # legend
    leg = ROOT.TLegend(0.7, 0.8, 0.9, 0.95)
    leg.SetFillStyle(0) # transparent background
    leg.SetBorderSize(0) # no border
    for h, label in zip(histograms, labels):
        leg.AddEntry(h, label, "l")
    leg.Draw()

    # saveplo
    os.makedirs("plots/comparison", exist_ok=True)
    
    c.SaveAs(f"plots/comparison/diphoton_{var}.pdf")
    c.SaveAs(f"plots/comparison/diphoton_{var}.png")
    c.SetLogy(1)
    c.Update()
    c.SaveAs(f"plots/comparison/diphoton_{var}_log.pdf")
    c.SaveAs(f"plots/comparison/diphoton_{var}_log.png")
    return None



# =====================
# CLI entrypoint
# =====================

def main():
    parser = argparse.ArgumentParser(description="Diphoton histogram utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract
    extract_parser = subparsers.add_parser("extract", help="Extract a histogram from a file")
    extract_parser.add_argument("--file", type=str, required=True, help="Histogram file path")

    # build
    build_parser = subparsers.add_parser("build", help="Build histogram data from a directory")
    build_parser.add_argument("--dir", type=str, required=True, help="Directory containing histogram files")
    build_parser.add_argument("--var", type=str, required=True, help="Variable name")

    # plot
    plot_parser = subparsers.add_parser("plot1", help="Plot two datasets")
    plot_parser.add_argument("--var", type=str, default="m34", help="Variable to plot")
    plot_parser.add_argument("--dir1", type=str, required=True, help="Directory for dataset")
    plot_parser.add_argument("--legend", type=str, required=False, help="Legend label for dataset")

    # compare
    compare_parser = subparsers.add_parser("plot", help="Overlay multiple datasets")
    compare_parser.add_argument("--var", type=str, required=True, help="Variable to plot")
    compare_parser.add_argument("--dirs", nargs="+", required=True, help="List of directories")
    compare_parser.add_argument("--labels", nargs="*", help="Labels for legend (must match number of dirs)")

    args = parser.parse_args()

    if args.command == "extract":
        data = extract_data(args.file)
        print(data)

    elif args.command == "build":
        data = arrange_data(args.dir, var=args.var)
        print(data)

    elif args.command == "plot1":
        plot1(args.dir1, var=args.var , lg = args.legend)

    elif args.command == "plot":
        plot(args.dirs, var=args.var, labels=args.labels)


if __name__ == "__main__":
    main()