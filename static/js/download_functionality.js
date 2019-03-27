function to_jupyter_notebook(entities, class_name, short_desc, long_desc){
    $.ajax({
        data: JSON.stringify(json),
        type: "POST",
        url: "/run_notebook",
        complete: function () {
            console.log("Result as JSON saved");
        }
    })
}