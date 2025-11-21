Inference Notebook
==================

The ``Inference.ipynb`` Jupyter notebook provides an interactive environment for running inference, benchmarking, and comparing PyTorch and TensorRT models.

Overview
--------

This notebook enables you to:

* Load both PyTorch and TensorRT models
* Run inference with both models on validation data
* Benchmark performance and compare latency
* Verify numerical correctness between models
* Analyze translation quality

Features
--------

Model Loading
~~~~~~~~~~~~~

The notebook loads:

* **PyTorch Model**: Trained transformer model from checkpoint files
* **TensorRT Engines**: Split engines (encoder, decoder, projection) from ``tensorrt_split/`` directory

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmarking function measures:

* **Latency Statistics**: Mean, p50 (median), p90, and p99 percentiles
* **Speedup Factor**: Ratio of PyTorch latency to TensorRT latency
* **CSV Export**: Results saved to ``benchmark_times.csv`` for analysis

The benchmark includes:

* Warmup runs to stabilize performance
* Multiple inference runs for statistical accuracy
* GPU synchronization for accurate timing

Output Comparison
~~~~~~~~~~~~~~~~~

The notebook includes several comparison functions:

* **Translation Comparison**: Side-by-side comparison of PyTorch vs TensorRT translations
* **Encoder Output Verification**: Numerical comparison of encoder outputs
* **Logits Comparison**: First-step logits comparison to verify decoder accuracy
* **Position Analysis**: Differences in padded vs unpadded positions

Usage
-----

1. **Start Jupyter Notebook**:

   .. code-block:: bash

      jupyter notebook Inference.ipynb

2. **Run all cells** sequentially to:

   * Load models and data
   * Run benchmarks
   * Compare outputs
   * Generate statistics

3. **View Results**:

   * Benchmark statistics are printed in the notebook
   * Translation examples show source, target, and predictions
   * CSV file contains detailed timing data

4. **TensorBoard Integration**:

   The notebook includes a cell to launch TensorBoard for viewing training logs.
   For remote access, use SSH port forwarding:

   .. code-block:: bash

      ssh -L 6005:localhost:6005 user@jetson_ip

   Then access TensorBoard at ``http://localhost:6005``

Expected Results
---------------

* **Speedup**: Typically 1.2x - 1.5x on Jetson devices
* **Numerical Accuracy**: Encoder output differences typically < 0.001
* **Translation Quality**: Both models should produce similar translations

Requirements
------------

* Jupyter notebook installed
* TensorRT engines built and available in ``tensorrt_split/``
* Trained model checkpoint available
* Validation dataset loaded

