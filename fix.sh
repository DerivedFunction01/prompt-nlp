git pull
jq -M 'del(.metadata.widgets)' dataset.ipynb > dataset2.ipynb 
mv dataset2.ipynb dataset.ipynb
git commit -a -m "Remove widgets from Colab"
git push