Usage
=====
After you installed TCAMpy and it's dependencies you can start using the model. First, you import the module, then create a 'TModel' class, like this:

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 5, 10)

The 'TModel' class takes a number of parameters. These determine the area, the different probabilities as well as the duration
of the model. See :ref:`api_docs` for further details about them. After you've created the class with the parameters of your
choice, you can simply run the entire model with one command:

.. code-block:: python

    M.run_model(plot = True, animate = False, stats = True)

This function automatically creates an initial state with a STC in the middle of the field. In every cycle, for the duration
of the model, it checks the entire area and chooses an action for each tumor cell. If 'plot' is set to True, the output includes multiple plots:
a growth plot, a cell number plot and a histogram of the proliferation potentails. If the 'stats' parameter is set to True,
various statistical properties of the model will be printed about the field and the cells. If 'animate' is set to True, the
function returns an animation of the growth. Note that animation does not work while using 'inline' backend. You must change
the backend first.

You can also modifiy the initial state by calling the following functions before running the model:

.. code-block:: python

    M.init_state()
    M.mod_cell(x, y, value)   # OR: M.field[y][x] = value

The 'mod_cell()' modifies the value of the cell at the (x, y) coordinates. For example, if you want to add another STC at (10,20),
use this command with x = 10, y = 20 and value = M.pmax + 1.

You can change the initial state as much as you'd like before running the model. For this function to work, first you need to create an initial state manually by the 'init_state()' function. You can remove the automatically
created STC by changing it's value to 0. (You don't need to call this function if you don't want to modifiy the initial state, running the model
creates a basic initial state, if you didn't define one before.)

You don't have to use this function to define a unique initial state; you can just make 'M.field' equal to a predefined numpy array or simply change it's value at a certain coordinate. Be careful to not put cells to the edge of the area. Here is
a complete example of a custom initial state, where the diagonal is filled with STCs (except on the edge).

.. code-block:: python

    import TCAMpy as tcam
    import numpy as np

    M = tcam.TModel(500, 50, 15, 1, 24, 1/24, 15, 4, 5, 10)

    M.field = (M.pmax+1) * np.eye(M.side_length)

    M.mod_cell(0, 0, 0)
    M.mod_cell(M.side_length-1, M.side_length-1, 0)

    M.run_model()

You can run multiple models using the 'run_multimodel()' function. You must specifiy how many models you'd like to run, and what initial state you'd
like to use for them as a numpy array. (If you want to stick with the default initial state just say 'M.field' without modifying it or give an empty numpy array.)
You can change 'M.field' or give a completely different numpy array as a parameter, if you want a different initial state. This function returns the results
as a pandas dataframe. Plot the averages with standard deviations by setting 'plot' to True and print the dataframe by setting 'stats' to True.

.. code-block:: python

    stats = M.run_multimodel(5, M.field, plot = True, stats = True)

Y can plot every individual model execution (after running 'run_model()' or 'run_multimodel()')
even if plotting was not enabled, because the data necessary for plotting is saved during every execution to M.runs.

.. code-block:: python

    # To plot execution number i of the model (i starts from 1 not from 0)
    M.plot_run(i)

    # Plot last execution:
    M.plot_run(len(M.runs))

    # To plot every previous execution
    for i in range(len(M.runs)):
        M.plot_run(i+1)

    # To clear previous execution data:
    M.runs = []

If you'd like to use this model on a graphical interface, you can create a streamlit dashboard (after creating model):

.. code-block:: python

    D = tcam.TDashboard(M)
    D.run_dashboard()

You will need to run the file containing this cod in your command line with streamlit. (If you are not in the directory of the file, define the path as well!)

.. code-block:: console

  streamlit run file_name.py

A dashboard will be created, where you have full control over the model. You can set the parameters using the sliders, run the model, view plots as well as statistics.
This dashboard can be used online with Streamlit Community Cloud, without any coding: https://tcampy.streamlit.app/

To use ML features provided by the module write the following code:

.. code-block:: python

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

You also have access to commands to save results (the field or the statistics) to an excel file, or create your own run function/loop by individually accessing cycles and cells. For details on those functions check the API Documentation.
