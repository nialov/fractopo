Gallery of fractopo example scripts and plots
=============================================

All matplotlib plots can be saved with:

.. code:: python

   fig.savefig("savename.png", bbox_inches="tight")
   # Or
   plt.savefig("savename.png", bbox_inches="tight")

Where ``savename`` can be replaced with name/path of where you
want to save the figure. ``bbox_inches`` is given to make sure the whole
plot is saved even thought individual elements go outside the ``matplotlib``
figure bounding box. ``png`` extension can be replaced with e.g. ``svg``.
See https://matplotlib.org/ for more information about plotting.
