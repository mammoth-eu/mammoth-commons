from mai_bias.backend.catalogue_loaders import registry


template_prefix = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        .main-content {
            margin-top: 60px;
        }
        .side-menu {
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            width: 200px;
            padding-top: 70px;
            background-color: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }
        .side-menu a {
            display: block;
            padding: 5px 20px;
            color: #495057;
            text-decoration: none;
        }
        .side-menu a:hover {
            background-color: #e9ecef;
        }
        .content-wrapper {
            margin-left: 220px;
            margin-right: 220px;
            padding: 20px;
        }
        h2 {
            color: #007bff;
            border-bottom: 3px solid #007bff;
            padding-bottom: 5px;
            margin-top: 20px;
        }
        
        /* Hide navbar on small screens */
        @media (max-width: 576px) {
            .side-menu {
                display: none;
            }
    
            .content-wrapper {
                margin-left: 0;
                margin-right: 0;
                padding: 0;
            }
            .main-content {
                margin-top: 200px;
            }
        }
    </style>
</head>
<body>
    <!-- Top Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div class="mx-auto">
            <ul class="navbar-nav">
                <li class="nav-item">
                    <a class="nav-link" href="index.html">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="datasets.html">Datasets</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="models.html">Models</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="analysis_methods.html">Analysis</a>
                </li>
            </ul>
        </div>
    </nav>

    <!-- Side Menu -->
    <div class="side-menu">
        {{ sidebar_content }}
    </div>

    <!-- Main Content -->
    <div class="content-wrapper">
        <div class="main-content container">
            <div class="container">
"""

template_postfix = """
            </div>
        </div>
    </div>

    <!-- Description Modal -->
    <div class="modal fade" id="descriptionModal" tabindex="-1" aria-labelledby="descriptionModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title text-muted" id="descriptionModalLabel">Description</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body" id="descriptionModalBody">
            <!-- Description will be injected here -->
          </div>
        </div>
      </div>
    </div>

<script>
    function showDescriptionModal(button) {
        const description = button.getAttribute("data-description");
        const name = button.getAttribute("data-name");
        document.getElementById("descriptionModalLabel").innerText = name;
        document.getElementById("descriptionModalBody").innerText = description;
        const modal = new bootstrap.Modal(document.getElementById("descriptionModal"));
        modal.show();
    }
</script>
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


def prepare(catalogue, page_title):
    sidebar_content = ""
    # sidebar_content = f"<h4 class='text-center'><a href='index.html'>Catalogue</a></h4>\n"

    # Main content with a page title
    main_content = f"<h1 class='display-4 text-center my-4'>{page_title}</h1>\n"

    # Loop through each loader and create a sidebar link and main section
    for loader in catalogue:
        loader_id = loader.replace(" ", "-").lower()
        sidebar_content += f"<a href='#{loader_id}'>{loader}</a>\n"
        main_content += f"<h2 id='{loader_id}'>{loader}</h2>\n"
        main_content += f"<p>{catalogue[loader]['description']}</p>\n"
        param_content = ""
        for param in catalogue[loader]["parameters"]:
            name, type, default, desc = param
            if desc == "Dataset" or desc == "Model" or desc == "Sensitive":
                continue
            param_content += f"{desc}\n"
        if param_content:
            main_content += "<b>Parameters</b><br>" + param_content + "<br>\n"
    return sidebar_content, main_content


# Generate pages with the sidebar and content
for page, (catalogue, page_title) in {
    "datasets.html": (registry.dataset_loaders, "Datasets"),
    "models.html": (registry.model_loaders, "Models"),
    "analysis_methods.html": (registry.analysis_methods, "Analysis"),
}.items():
    sidebar_content, main_content = prepare(catalogue, page_title)
    full_content = (
        template_prefix.replace("{{ title }}", page_title).replace(
            "{{ sidebar_content }}", sidebar_content
        )
        + main_content
        + template_postfix
    )
    with open(f"docs/{page}", "w") as file:
        file.write(full_content)

index_content = """
<h1 class='display-4 text-center my-4'>MAMMOth Catalogue</h1>
<p class='lead text-center'>Explore the core modules of the MAI-BIAS toolkit and demonstrator.</p>
<p>This catalogue gathers the MAI-BIAS modules developed by the <a href=https://mammoth-ai.eu/>MAMMOth</a> project.
These hold research results and third-party libraries to perform various kinds of fairness assessment.
Some will also guide you on how to apply mitigation strategies. 
The documentation found here is the same as the one shown by the respective tools. However, by organizing
everything in one place, it becomes easier to understand all available options for the toolkit.
Use the navigation links to overview datasets, models, and analysis methods.</p>
<p>Broadly, we offer three types of modules:</p>
<div class="container my-4">
    <div class="row">
        <!-- Dataset Loaders Card -->
        <div class="col-md-4">
            <div class="card text-center shadow-sm">
                <div class="card-body">
                    <h5 class="card-title">🗄 <a href='datasets.html'>Dataset loaders</a></h5>
                    <p class="card-text">Load your own datasets or automatically use public datasets that are popular in fairness literature research.</p>
                </div>
            </div>
        </div>

        <!-- Model Loaders Card -->
        <div class="col-md-4">
            <div class="card text-center shadow-sm">
                <div class="card-body">
                    <h5 class="card-title">🧮 <a href='models.html'>Model loaders</a></h5>
                    <p class="card-text">Import a wide range of trained machine learning model formats or other artificial intelligence algorithms.</p>
                </div>
            </div>
        </div>

        <!-- Analysis Metrics Card -->
        <div class="col-md-4">
            <div class="card text-center shadow-sm">
                <div class="card-body">
                    <h5 class="card-title">🔬 <a href='analysis_methods.html'>Analysis metrics</a></h5>
                    <p class="card-text">Includes traditional metrics quantifying disparities between sensitive groups and visualizations that aid explainability.</p>
                </div>
            </div>
        </div>
    </div>
</div>


<p>Visit our <a href="https://github.com/mammoth-eu/mammoth-commons">GitHub</a> repository
to quickly set up a demonstrator that lets you run all these modules.<p>
"""

with open("docs/index.html", "w") as file:
    file.write(
        template_prefix.replace("{{ title }}", "MAMMOth Catalogue").replace(
            "{{ sidebar_content }}", ""
        )
        + index_content
        + template_postfix
    )
