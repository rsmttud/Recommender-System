$(document).ready(function () {
    let clustering_txt_short = "I want to find pattern in my machine errors";
    let clustering_txt_long = "I want to find pattern in my machine errors";

    $("#short_description").val(clustering_txt_short);
    $("#long_description").val(clustering_txt_long);
    $("#title").val("Test1");


     $("#points1").mouseover(function(){
           $("#points1").find("path").css("transform", "scale(1.15)");
           console.log("mouseover!")
     })

});