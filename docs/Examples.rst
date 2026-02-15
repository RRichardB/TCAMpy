Example codes
=============

Single model
------------

Create a model and execute it one time. Plot the results, animate and show the statistics. (Animation doesn't work with inline backend)

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 5, 10)
    M.run_model(plot = True, animate = True, stats = True)

Multiple executions
-------------------

Run the model multiple (in this case 5) times and plot the average results.

.. code-block:: python

    import TCAMpy as tcam
    import matplotlib.pyplot as plt

    M = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 5, 10)
    stats = M.run_multimodel(5, M.field, plot = True, stats = True)

    # Check visualization for every execution
    for i in range(len(M.runs)):
        M.plot_run(i+1)

Modifying initial state
-----------------------

Modify the initial state by defining/changing M.field or calling 'mod_cell()' (can only be called after M.field was created either manually or by 'init_state()').

.. code-block:: python

    import TCAMpy as tcam
    import numpy as np

    M = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 5, 10)

    M.field = (M.pmax+1) * np.eye(M.side_length)
    M.field[0][0] = 0
    M.mod_cell(M.side_length-1, M.side_length-1, 0)

    M.run_model()

Dashboard
---------

Create a streamlit dashboard. Run the file containing the code with streamlit to open the application.

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 5, 10)

    D = tcam.TDashboard(M)
    D.run_dashboard()

.. code-block:: console

  streamlit run file_name.py

This dashboard can be used online with Streamlit Community Cloud, without any coding: https://tcampy.streamlit.app/

Machine Learning
----------------

Use ML features to generate dataset, train a model and predict tumor size based on given parameters

.. code-block:: python

    import TCAMpy as tcam
    
    M  = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 4, 10)
    ml = tcam.TML(M)
    
    # Select parameters to randomize and ranges
    randomize = {"PA": (1, 15), "M": (0, 10), "I": (0, 10)}
    
    # Generate dataset with randomized parameters
    df = ml.generate_dataset(
        n=50,
        random_params=randomize,
        output_file="tumor_dataset.csv"
    )
    
    # Train a model
    model, metrics = ml.train_predictor("tumor_dataset.csv", "Tumor size")
    
    new_params = [500, 50, 15, 1, 24, 1/24, 15, 4, 4, 10]
    print ("Predicted Attribute: ", ml.predict_new(new_params))

Example files
-------------

For example files please visit: https://github.com/Fetasalyt/TCAMpy/tree/main/examples
