
echo ">>> Creating figures in 'analysis/figures'."

echo ">>> Catchment visualization (fig01.png)."
python analysis/catchment_vis.py

echo ">>> Model visualization (fig04.png + fig05.png)."
python analysis/model_vis.py

echo ">>> Timeseries visualization (fig06.png)."
python analysis/timeseries_vis.py

echo ">>> Annual visualization (fig07.png)."
python analysis/annual_vis.py

echo ">>> Annual visualization (fig08.png + fig09.png)."
python analysis/upscaling_vis.py

echo ">>> HP tuning visualization (figA01.pdf + figA02.pdf)."
python analysis/hptuning_vis.py

echo ">>> Done!."