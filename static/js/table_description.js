function implant_table(entities, table_length) {
    //Table Head
    if (entities.length > 0) {
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
                $("#data-structure-table tbody").find(id).append("<td>" + entity + "_" + row + "</td>")
            });
        });
    }else{
         $("#data-structure-table").append("<p>Sorry a recommendation for the data structure is only" +
             " possible with extracted entities.</p>");
    }

    //Set Line-Height of Text
    let height = $("#data-structure-table").height()+"px";
    $("#data-structure-description > h5").css("line-height", height);

}