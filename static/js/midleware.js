//TODO long desc, short, desc should be delivered as well
$("document").ready(function () {
    $("#submit-button").click(function () {
        $.ajax({
            data: {
                short_description: $("#short_description").val(),
                long_description: $("#long_description").val(),
                title: $("#title").val(),
                method: $("#method").val()
            },
            type: "POST",
            url: "/classify",
            beforeSend: function () {
                console.log("tabs swiped");
                //Swipe to next tab
                $("#nav-input").removeClass("active");
                $("#nav-process").addClass("active");
                $("#tabs-swipe-demo").tabs({swipeable: true});
            },
            success: function (data) {
                // Swipe to next tab
                $("#tabs-swipe-demo").tabs({swipeable: false});
                $("#nav-input").removeClass("active");
                $("#nav-process").removeClass("active");
                $("#nav-result").addClass("active");
                $("#tabs-swipe-demo").tabs({swipeable: true});
                console.log("tabs swiped again k.");
                //$("#tabs-swipe-demo").tabs({swipeable: true});

                //Remove all UL under #output-result-main
                $("#output-result-main > ul").remove();

                // Writing input in result page
                let ul = $("<ul/>").appendTo($("#output-result-main"));
                $.each(data, function (key, value) {
                    $('<li/>')
                        .attr('role', 'menuitem')
                        .text(key + ": " + value)
                        .appendTo(ul)
                });
            }
        })
    });
});

//Returns String
function createOutputJSON(long_desc, short_desc, dict_result_long, dict_result_short){
    //Transforming to JSON
    /*
    {
        short_desc: {
            message:
            classes:
            probabilities:
        }
        long_desc: {
            message:
            classes:
            probabilities:
        }
    }

     */
}
//Save File
function saveJSONString(file_name, json_string){
    //Save incoming JSON String to file
}
