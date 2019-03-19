function implant_class_definitions(class_name) {

    var path = get_path_to_txt(class_name);
    var class_name = class_name.charAt(0).toUpperCase() + class_name.slice(1);
    $.get(path, function (data) {
        $("#analysis-definition")
            .append("<h4>" + class_name + " Definition: </h4>")
        $("#analysis-definition")
            .append("<span>" + data + "</span>")
    }, "text")
}

function get_path_to_txt(class_name) {
    var path = "static/txt/placeholder.txt";

    // TODO create Placeholder pic and other charts
    if (class_name === "clustering") {
        path = "static/txt/clustering.txt";
    } else if (class_name === "classification") {
        path = "static/txt/classification.txt";
    } else if (class_name === "regression") {
        path = "static/txt/classification.txt";
    } else if (class_name === "pattern_mining") {
        path = "static/txt/pattern_mining.txt";
    }

    return path
}