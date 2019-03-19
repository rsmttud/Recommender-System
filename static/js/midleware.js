/*TODO
 * Cleanup Code (It's a mess),
 * More Encapsulation in Function,
 * Store long_desc, short_desc, classification_results for both, combined_classification result and final fore in json
 * Save Json with timestamp (Can be used to retrain the model later in a semi-supervised way for example)
 * Load Json for Donout Chart or other descriptive stuff
 *
  * */
$("document").ready(function () {
    $("#input-form").submit(function (e) {
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
            beforeSend: function () {
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

                //Swipe to result page
                $("#nav-result").parent("li").removeClass("disabled");
                $("#nav-download").parent("li").removeClass("disabled");
                $("#tabs-swipe-demo").tabs({swipeable: false});
                //Remove html tags before writing new information to them
                reset_html();
                //Fire modal with entities
                var most_prob_class = Object.keys(data["forecast"]).reduce(function (a, b) {
                    return data["forecast"][a] > data["forecast"][b] ? a : b
                });
                implant_modal_functionality(data["entities"], most_prob_class);
                $('#modal-entity').modal('open');

                $("#modal-entity-submit").click(function () {
                    // Write most probable class to html
                    // Get key with highest value from dict
                    let entities = [];
                    $("input[checked='checked']").each(function (index, element) {

                        if ($(element).is(':checked')) {
                            entities.push($(element).next("span").text())
                        }

                    });

                    if (most_prob_class === "prediction") {
                        if ($("#modal-content-radio-yes").is(':checked')) {
                            initialize_result_page(data, entities, "regression")
                        } else {
                            console.log("Radio Button not checked (Classification)");
                            initialize_result_page(data, entities, "classification")
                        }
                    }else{
                        initialize_result_page(data, entities, most_prob_class)
                    }

                });


            }
        })

    });
});

function get_json(forecast, entities) {
    let short_desc = $('#short_description').val(),
        long_desc = $("#long_description").val(),
        title = $("#title").val(),
        //TODO Needs to be adjusted for final prototyp
        method = $("#method").val(); // The method from the select menue

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
        forecast: forecast,
        entities: entities
    };

    return _json
}

function send_json_to_python_backend(json) {
    $.ajax({
        data: JSON.stringify(json),
        contentType: 'application/json',
        dataType: "json",
        type: "POST",
        url: "/save_json",
        complete: function () {
            console.log("Result as JSON saved");
        }
    })
}

function reset_html() {
    console.log("reset called");
    $("#output-result-main-class h5").remove();
    $("#donut-chart").empty();
    $("#modal-form").empty();
}

function implant_modal_functionality(entities, most_likely_class) {
    if (most_likely_class == "prediction") {
        console.log(most_likely_class);
        $("#modal-content-prediction").css("display", "block")
    } else {
        console.log(most_likely_class)
    }

    entities.forEach(function (entity) {
        let _html = "<p><label><input type=\"checkbox\" id=\"test\" checked=\"checked\"/><span>" + entity + "</span></label>\</p>";
        $("#modal-form").append(_html)
    });
}

function initialize_result_page(data, entities, class_name) {


    $("#output-result-main-class")
        .append("<h5>" + class_name.charAt(0).toUpperCase() + class_name.slice(1) + "</h5>");

    //<li class = "collection-item">Lorem Ipsum</li>
    entities.forEach(function (element) {
        $("#output-result-main-features > ul").append("<li class = \"collection-item\">" + element + "</li>")
    });
    // Preparing the data for donout_chart.js
    let forecast = [];
    $.each(data["forecast"], function (key, value) {
        forecast.push({"class": key, "prob": value})
    });

    let json = get_json(forecast, data["entities"]);
    console.log(json);
    donutChart(json);
    send_json_to_python_backend(json);
    implant_analysis_chart(class_name, entities);
    implant_class_definitions(class_name);
    implant_table(entities, 4)
}

/*This is lazy workaround*/
$("document").ready(function () {
   $("#nav-input").click(function (){
       location.reload();
   } )
});