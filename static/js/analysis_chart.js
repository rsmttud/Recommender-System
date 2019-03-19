function implant_analysis_chart(class_name, entities) {

    path = get_path_to_img(class_name);

    if(entities.length <=1){
        entities = ["Entity 1", "Entity 2"]
    }

    // TODO Functionallity when file not found
    $.get(path, function (data) {
        $("#analysis-chart").append(data.documentElement)
            .find("#entity-placeholder-1")
            .text(entities[0]);
        $("#analysis-chart").find("#entity-placeholder-2")
            .text(entities[1])
    })
}

function get_path_to_img(class_name) {
    var path = "static/img/analysis_chart/placeholder_placeholder.svg";

    // TODO create Placeholder pic and other charts
    if (class_name === "clustering") {
        path = "static/img/analysis_chart/clustering_placeholder.svg";
    } else if (class_name === "classification") {
        path = "static/img/analysis_chart/classification_placeholder.svg";
    } else if (class_name === "regression") {
        path = "static/img/analysis_chart/regression_placeholder.svg";
    }else if (class_name === "pattern_mining") {
        path = path;
    }

    return path
}
