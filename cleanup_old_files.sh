#!/bin/bash
# Script to remove old files that are no longer needed

echo "Files that can be removed (replaced by modular structure):"
echo ""
echo "1. Old Jupyter Notebooks:"
echo "   - 01-Data.ipynb"
echo "   - 02-Model.ipynb"
echo "   - 03-Train.ipynb"
echo ""
echo "2. Old model file:"
echo "   - model.py"
echo ""
echo "These files have been replaced by:"
echo "   - src/data/dataset.py"
echo "   - src/models/unet.py"
echo "   - src/training/trainer.py"
echo "   - scripts/train.py"
echo "   - scripts/test.py"
echo ""
read -p "Do you want to remove these files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing old files..."
    rm -f 01-Data.ipynb
    rm -f 02-Model.ipynb
    rm -f 03-Train.ipynb
    rm -f model.py
    echo "Done! Old files removed."
else
    echo "Files kept. You can remove them manually later."
fi

