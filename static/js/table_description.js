function implant_table(class_name, entities, table_length) {
    if (class_name === "pattern_mining") {
        set_pattern_mining_table(entities)
    } else if (class_name === "clustering") {
        set_clustering_table(entities)
    } else if (class_name === "regression") {
        set_regression_table(entities, table_length)
    } else if (class_name === "classification") {
        set_classification_table(entities, table_length)
    }

    set_line_height()

}

function set_pattern_mining_table(entities, table_length = 4) {
    entities = entities.slice(0, 4);

    let e_placeholder = ["Entity<sub>1</sub>", "Entity<sub>2</sub>", "...", "Entity<sub>4</sub>"];
    if (entities.length === 0) {
        entities = e_placeholder
    } else if (entities.length >= 1 && entities.length <= 4) {
        entities.slice(0, entities.length).forEach(function (e, i) {
            e_placeholder[i] = e
        });
        entities = e_placeholder
    } else {
        entities = entities.slice(0, 4)
    }


    $("#data-structure-table").append("<thead><tr></tr></thead>");
    $("#data-structure-table tr").append("<th>time</th>");
    entities.forEach(function (entity) {
        $("#data-structure-table tr").append("<th>" + entity + "</th>")
    });

    //Table Body
    let range = n => Array.from(Array(table_length).keys()); // range function
    $("#data-structure-table").append("<tbody></tbody>");

    range(table_length).forEach(function (row) {
        let date = parseInt(row) + 1 + "-08-2018";
        let id = "#" + row + "-tr";
        $("#data-structure-table tbody").append("<tr id =\"" + row + "-tr" + "\">");
        $("#data-structure-table tbody").find(id).append("<td>" + date + "</td>");


        entities.forEach(function (entity) {
            let id = "#" + row + "-tr";
            $("#data-structure-table tbody").find(id).append("<td>...</td>")
        });
    });


}

function set_clustering_table(entities) {
    if (entities.length < 1) {
        entities = ["Group"]
    }
    let columns = ["Entity", "C<sub>1</sub>", "C<sub>2</sub>", "...", "C<sub>n</sub>"];

    $("#data-structure-table").append("<thead><tr></tr></thead>");
    columns.forEach(function (column, index) {

        $("#data-structure-table tr").append("<th>" + column + "</th>")
    });

    //Table Body
    let range = n => Array.from(Array(4).keys()); // range function
    $("#data-structure-table").append("<tbody></tbody>");
    entities = [entities[0] + "<sub>A</sub>", entities[0]+"<sub>B</sub>", entities[0] + "<sub>C</sub>", "...", entities[0]+"<sub>N</sub>"];
        entities.forEach(function (entity, index) {
            let id = "#" + index + "-tr";
            $("#data-structure-table tbody").append("<tr id =\"" + index + "-tr" + "\">");
            $("#data-structure-table tbody").find(id).append("<td>" + entity + "</td>");
            range(columns.length).forEach(function () {
                $("#data-structure-table tbody").find(id).append("<td>...</td>");
            })
        });


}

function set_regression_table(entities, table_length = 4) {
    let e_placeholder = ["Entity<sub>1</sub>", "Entity<sub>2</sub>", "...", "Entity<sub>4</sub>"];
    if (entities.length === 0) {
        entities = e_placeholder
    } else if (entities.length >= 1 && entities.length <= 4) {
        entities.slice(0, entities.length).forEach(function (e, i) {
            e_placeholder[i] = e
        });
        entities = e_placeholder
    } else {
        entities = entities.slice(0, 4)
    }
    //Table Head
    $("#data-structure-table").append("<thead><tr></tr></thead>");
    entities.forEach(function (entity) {
        $("#data-structure-table tr").append("<th>" + entity + "</th>")
    });
    //Table Body
    let range = n => Array.from(Array(table_length).keys());
    $("#data-structure-table").append("<tbody></tbody>");
    range(table_length).forEach(function (row) {
        $("#data-structure-table tbody").append("<tr id =\"" + row + "-tr" + "\">");
        entities.forEach(function (entity) {
            let id = "#" + row + "-tr";
            $("#data-structure-table tbody").find(id).append("<td>...</td>")
        });
    });
}

function set_classification_table(entities, table_length = 4) {
//Table Head
    let entity = "Target";
    if (entities.length > 1) {
        entity = entities[0]
    }
    let columns = ["C<sub>1</sub>", "C<sub>2</sub>", "...", "C<sub>n</sub>", entity];
    $("#data-structure-table").append("<thead><tr></tr></thead>");
    columns.forEach(function (columns) {
        $("#data-structure-table tr").append("<th>" + columns + "</th>")
    });

    //Table Body
    let range = n => Array.from(Array(table_length).keys());
    $("#data-structure-table").append("<tbody></tbody>");
    range(table_length).forEach(function (row) {
        $("#data-structure-table tbody").append("<tr id =\"" + row + "-tr" + "\">");
        columns.forEach(function (column, index) {
            let id = "#" + row + "-tr";
            if (index !== columns.length) {
                $("#data-structure-table tbody").find(id).append("<td>...</td>")
            } else {
                $("#data-structure-table tbody").find(id).append("<td>" + entity + "</td>")
            }

        });
    });

}

function set_line_height() {
    //Set Line-Height of Text
    let height = $("#data-structure-table").height() + "px";
    $("#data-structure-description > h5").css("line-height", height);
}
