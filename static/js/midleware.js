
/*TODO
 * Cleanup Code (It's a mess),
 * More Encapsulation in Function,
 * Store long_desc, short_desc, classification_results for both, combined_classification result and final fore in json
 * Save Json with timestamp (Can be used to retrain the model later in a semi-supervised way for example)
 * Load Json for Donout Chart or other descriptive stuff
 *
  * */
$("document").ready(function () {
    $("#input-form").submit(function(e) {
        e.preventDefault();
        $.ajax({
            data: {
                short_description: $('#short_description').val(),
                long_description: $("#long_description").val(),
                title: $("#title").val(),
                method: $("#method").val()
            },
            type: "POST",
            url: "/classify",
            beforeSend: function() {

                //Swipe to next tab
                $("#nav-input").removeClass("active");
                $("#nav-input").parent("li").addClass("disabled");
                $("#nav-process").addClass("active");
                $("#nav-process").parent("li").removeClass("disabled");
                $("#tabs-swipe-demo").tabs({swipeable: false});
            },
            success: function (data) {

                //Some CSS adjustments
                $("#nav-input").parent("li").removeClass("disabled");

                $("#nav-process").removeClass("active");
                $("#nav-process").parent("li").addClass("disabled");
                $("#nav-result").addClass("active");
                $("#nav-result").parent("li").removeClass("disabled");

                $("#nav-download").parent("li").removeClass("disabled");
                $("#tabs-swipe-demo").tabs({swipeable: false});

                //Remove html tags before writing new information to them
                reset_html();


                // Write most probable class to html
                // Get key with highest value from dict
                var most_prob_class = Object.keys(data).reduce(function(a, b){ return data[a] > data[b] ? a : b });
                // Writing input in result page
                $("#output-result-main-class")
                    .append("<h6>"+most_prob_class);


                // Preparing the data for donout_chart.js
                var forecast = [];
                $.each(data, function(key, value){
                    forecast.push({"class": key, "prob": value})
                });

                var _json = get_json(forecast);
                donoutChart(_json);
                // Save output to json
                send_json_to_python_backend(_json);
                console.log(_json)
            }
        })

    });
});

function get_json(forecast){
    let short_desc = $('#short_description').val(),
        long_desc =  $("#long_description").val(),
        title = $("#title").val(),
        method = $("#method").val();

    let date = new Date();
    let timestamp = date.getTime();

    let _json = {
        title: title,
        short_desc: {
            text: short_desc,
            probs: []

        },
        long_desc: {
            text: long_desc,
            probs: []

        },
        method: method,
        forecast: forecast
    };

    return _json
}

function send_json_to_python_backend(json_str){
     $.ajax({
         data: JSON.stringify(json_str),
         type: "POST",
         url: "/save_json",
         complete: function(data){
             console.log("Result as JSON saved: "+data);
         }
     })
}

function reset_html(){
    $("#output-result-main > h5").remove();
    $("#donut-chart > svg").remove();
}