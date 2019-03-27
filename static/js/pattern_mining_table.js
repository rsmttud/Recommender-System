function build_pattern_mining_table(entities) {
    $("#analysis-chart").append("<table class = \"highlight centered\"><thead><tr></tr></thead></table>");

    let header_table = ["#", "Pattern", "Confidence"];
    header_table.forEach(function (e) {
        $("#analysis-chart tr").append("<th>" + e + "</th>")
    });

    let patterns = generate_example_pattern(entities);
    $("#analysis-chart table").append("<tbody></tbody>");
    patterns.forEach(function (d, i) {
        let td_1 = "<td>"+i+1+"</td>";
        let td_2 =  "<td><b>"+d+"</b></td>";
        let td_3 =  "<td>"+(85-i*(Math.random()+1)).toFixed(2)+"% </td>"; // Number 85 and decreasing

        $("#analysis-chart tbody").append("<tr>"+td_1+td_2+td_3+"</tr>")
    });

     $("#analysis-chart").append("<div class = \"right-align\"><i>entity x, entity y: other potential entities in your data.</i></div>")


}

function generate_example_pattern(entities) {
    let placeholder = ["{entity x, entity y} →", "entity x} → {entity y}"];
    let patterns = [];
    entities = entities.slice(0,2); // only work with the two first entities
    entities.forEach(function (d, i) {
         patterns.push(placeholder[0] + " " + "{"+d+"}");
         patterns.push("{"+d+ ", " + placeholder[1]);
         if(i === 0){
             patterns.push("{entity z, entity y} → "+ "{"+d+"}")
         }


    });

    return patterns
}