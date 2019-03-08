
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
                $("#nav-input").parent("li").removeClass("disabled");

                $("#nav-process").removeClass("active");
                $("#nav-process").parent("li").addClass("disabled");
                $("#nav-result").addClass("active");
                $("#nav-result").parent("li").removeClass("disabled");

                $("#nav-download").parent("li").removeClass("disabled");
                $("#tabs-swipe-demo").tabs({swipeable: false});

                //Remove html tags before writing new information to them
                $("#output-result-main > h5").remove();
                $("#donut-chart > svg").remove();


                // Get key with highest value from dict
                var most_prob_class = Object.keys(data).reduce(function(a, b){ return data[a] > data[b] ? a : b });

                // Writing input in result page
                $("#output-result-main-class")
                    .append("<h6>"+most_prob_class);

                // Preparing the data for donout_chart.js
                var lst = [];
                $.each(data, function(key, value){
                    lst.push({"class": key, "prob": value})
                });

                var _json = {forecast: lst};
                //Path where json is stored..
                donoutChart(_json);
                console.log(data)
            }
        })

    });
});