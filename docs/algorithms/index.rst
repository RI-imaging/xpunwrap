Algorithm Reference
===================

Line-by-line mathematical descriptions of each algorithm.

Overview
--------

.. list-table:: Algorithm overview
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Algorithm
     - Speed
     - GPU friendly
     - Handles discontinuity
     - References
   * - LS Poisson (:func:`unwrap_phase_gpu.algorithms.algo_ls_poisson`)
     - very fast
     - excellent
     - poor
     - :ref:`Ghiglia & Pritt (1998) <ls_poisson_refs>`
   * - | LS Poisson
       | (Periodic Grad) (:func:`unwrap_phase_gpu.algorithms.algo_ls_poisson_periodic_grad`)
     - very fast
     - excellent
     - poor
     - :ref:`Ghiglia & Pritt (1998) <ls_poisson_periodic_grad_refs>`
   * - Weighted LS (:func:`unwrap_phase_gpu.algorithms.algo_ls_weighted`)
     - medium
     - good
     - moderate
     - :ref:`D. C. Ghiglia & L. A. Romero (1994) <ls_weighted_refs>`, :ref:`Ghiglia & Pritt (1998) <ls_weighted_refs>`
   * - TV-L1 (:func:`unwrap_phase_gpu.algorithms.algo_tvl1`)
     - slow
     - good
     - excellent
     - :ref:`A. Chambolle & T. Pock (2011) <tvl1_refs>`
   * - | Reliability Path
       | (**Not Implemented**)
     - medium
     - poor
     - excellent
     - `skimage example <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_.

.. toctree::
   :maxdepth: 1

   ls_poisson
   ls_poisson_periodic_grad
   ls_weighted
   tvl1
