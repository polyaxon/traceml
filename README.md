[![License: Apache 2](https://img.shields.io/badge/License-apache2-green.svg)](LICENSE)
[![TraceML](https://github.com/polyaxon/traceml/actions/workflows/traceml.yml/badge.svg)](https://github.com/polyaxon/traceml/actions/workflows/traceml.yml)
[![Slack](https://img.shields.io/badge/chat-on%20slack-aadada.svg?logo=slack&longCache=true)](https://polyaxon.com/slack/)
[![Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg?style=flat)](https://polyaxon.com/docs/)
[![GitHub](https://img.shields.io/badge/issue_tracker-github-blue?logo=github)](https://github.com/polyaxon/polyaxon/issues)
[![GitHub](https://img.shields.io/badge/roadmap-github-blue?logo=github)](https://github.com/polyaxon/polyaxon/milestones)

<br>
<p align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/polyaxon/polyaxon/master/artifacts/packages/traceml.svg" alt="traceml" height="100">
  </p>
</p>
<br>

# TraceML

Engine for ML/Data tracking, visualization, and dashboards for Polyaxon.

## Install

```bash
pip install traceml
```

If you would like to use the tracking features, you need to install `polyaxon` as well:

```bash
pip install polyaxon traceml
```

## Local sandbox

> WIP: this command is in preview

Start a local sandbox to track and visualize the run

```bash
polyaxon sandbox -f path/to/artifacts/repo
```

## Offline usage

You can enable the offline mode to track runs without an  API:

```bash
export POLYAXON_OFFLINE="true"
```

Or passing the offline flag

```python
from traceml import tracking

tracking.init(..., is_offline=True, ...)
```

## Simple usage in a Python script

```python
import random

import traceml as tracking

tracking.init(
    is_offline=True,
    project='quick-start',
    name="my-new-run",
    description="trying TraceML",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
) 

# Tracking some data refs
tracking.log_data_ref(content=X_train, name='x_train')
tracking.log_data_ref(content=y_train, name='y_train')

# Tracking inputs
tracking.log_inputs(
    batch_size=64,
    dropout=0.2,
    learning_rate=0.001,
    optimizer="Adam"
)

def get_loss(step):
    result = 10 / (step + 1)
    noise = (random.random() - 0.5) * 0.5 * result
    return result + noise

# Track metrics
for step in range(100):
    loss = get_loss(step)
    tracking.log_metrics(
    loss=loss,
    accuracy=(100 - loss) / 100.0,
)

# Track some one time results
tracking.log_outputs(validation_score=0.66)

# Optionally manually stop the tracking process
tracking.stop()
```

## Integration with deep learning and machine learning libraries and frameworks

### Keras

You can use TraceML's callback to automatically save all metrics and collect outputs and models, you can also track additional information using the logging methods:

```python
from traceml import tracking
from traceml.integrations.keras import Callback

tracking.init(
    is_offline=True,
    project='tracking-project',
    name="keras-run",
    description="trying TraceML & Keras",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
)

tracking.log_inputs(
    batch_size=64,
    dropout=0.2,
    learning_rate=0.001,
    optimizer="Adam"
)
tracking.log_data_ref(content=x_train, name='x_train')
tracking.log_data_ref(content=y_train, name='y_train')
tracking.log_data_ref(content=x_test, name='x_test')
tracking.log_data_ref(content=y_test, name='y_test')

# ...

model.fit(
    x_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=100,
    callbacks=[Callback()],
)
```

### PyTorch

You can log metrics, inputs, and outputs of Pytorch experiments using the tracking module: 

```python
from traceml import tracking

tracking.init(
    is_offline=True,
    project='tracking-project',
    name="pytorch-run",
    description="trying TraceML & PyTorch",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
)

tracking.log_inputs(
    batch_size=64,
    dropout=0.2,
    learning_rate=0.001,
    optimizer="Adam"
)

# Metrics
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    tracking.log_mtrics(loss=loss)
    
asset_path = tracking.get_outputs_path('model.ckpt')
torch.save(model.state_dict(), asset_path)

# log model
tracking.log_artifact_ref(asset_path, framework="pytorch", ...)
```

### Tensorflow

You can log metrics, outputs, and models of Tensorflow experiments and distributed Tensorflow experiments using the tracking module:

```python
from traceml import tracking
from traceml.integrations.tensorflow import Callback

tracking.init(
    is_offline=True,
    project='tracking-project',
    name="tf-run",
    description="trying TraceML & Tensorflow",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
)

tracking.log_inputs(
    batch_size=64,
    dropout=0.2,
    learning_rate=0.001,
    optimizer="Adam"
)

# log model
estimator.train(hooks=[Callback(log_image=True, log_histo=True, log_tensor=True)])
```

### Fastai

You can log metrics, outputs, and models of Fastai experiments using the tracking module:

```python
from traceml import tracking
from traceml.integrations.fastai import Callback

tracking.init(
    is_offline=True,
    project='tracking-project',
    name="fastai-run",
    description="trying TraceML & Fastai",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
)

# Log model metrics
learn.fit(..., cbs=[Callback()])
```

### Pytorch Lightning

You can log metrics, outputs, and models of Pytorch Lightning experiments using the tracking module:

```python
from traceml import tracking
from traceml.integrations.pytorch_lightning import Callback

tracking.init(
    is_offline=True,
    project='tracking-project',
    name="pytorch-lightning-run",
    description="trying TraceML & Lightning",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
)

...
trainer = pl.Trainer(
    gpus=0,
    progress_bar_refresh_rate=20,
    max_epochs=2,
    logger=Callback(),
)
```

### HuggingFace

You can log metrics, outputs, and models of HuggingFace experiments using the tracking module:

```python
from traceml import tracking
from traceml.integrations.hugging_face import Callback

tracking.init(
    is_offline=True,
    project='tracking-project',
    name="hg-run",
    description="trying TraceML & HuggingFace",
    tags=["examples"],
    artifacts_path="path/to/artifacts/repo"
)

...
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    callbacks=[Callback],
    # ...
)
```

## Tracking artifacts

```python
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from bokeh.plotting import figure
from vega_datasets import data

from traceml import tracking


def plot_mpl_figure(step):
    np.random.seed(19680801)
    data = np.random.randn(2, 100)

    figure, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data[0])
    axs[1, 0].scatter(data[0], data[1])
    axs[0, 1].plot(data[0], data[1])
    axs[1, 1].hist2d(data[0], data[1])

    tracking.log_mpl_image(figure, 'mpl_image', step=step)


def log_bokeh(step):
    factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
    x = [50, 40, 65, 10, 25, 37, 80, 60]

    dot = figure(title="Categorical Dot Plot", tools="", toolbar_location=None,
                 y_range=factors, x_range=[0, 100])

    dot.segment(0, factors, x, factors, line_width=2, line_color="green", )
    dot.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

    factors = ["foo 123", "bar:0.2", "baz-10"]
    x = ["foo 123", "foo 123", "foo 123", "bar:0.2", "bar:0.2", "bar:0.2", "baz-10", "baz-10",
         "baz-10"]
    y = ["foo 123", "bar:0.2", "baz-10", "foo 123", "bar:0.2", "baz-10", "foo 123", "bar:0.2",
         "baz-10"]
    colors = [
        "#0B486B", "#79BD9A", "#CFF09E",
        "#79BD9A", "#0B486B", "#79BD9A",
        "#CFF09E", "#79BD9A", "#0B486B"
    ]

    hm = figure(title="Categorical Heatmap", tools="hover", toolbar_location=None,
                x_range=factors, y_range=factors)

    hm.rect(x, y, color=colors, width=1, height=1)

    tracking.log_bokeh_chart(name='confusion-bokeh', figure=hm, step=step)


def log_altair(step):
    source = data.cars()

    brush = alt.selection(type='interval')

    points = alt.Chart(source).mark_point().encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color=alt.condition(brush, 'Origin:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )

    bars = alt.Chart(source).mark_bar().encode(
        y='Origin:N',
        color='Origin:N',
        x='count(Origin):Q'
    ).transform_filter(
        brush
    )

    chart = points & bars

    tracking.log_altair_chart(name='altair_chart', figure=chart, step=step)


def log_plotly(step):
    df = px.data.tips()

    fig = px.density_heatmap(df, x="total_bill", y="tip", facet_row="sex", facet_col="smoker")
    tracking.log_plotly_chart(name="2d-hist", figure=fig, step=step)


plot_mpl_figure(100)
log_bokeh(100)
log_altair(100)
log_plotly(100)
```
