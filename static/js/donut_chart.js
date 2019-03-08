function donoutChart(data) {
    data = data["forecast"]; //Place in dict where final prediction values are stored
    console.log(data);
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
        width = 400 - margin.right - margin.left,
        height = 400 - margin.top - margin.bottom,
        radius = width / 2.5;

    var color = d3.scaleOrdinal()
        .range(["#b4126b", "#007581", "#055E67", "#C0AD26"]); //Color range
    // Arc Generator produces the arcs for the pie generator
    var arc = d3.arc()
        .outerRadius(radius - 10)
        .innerRadius(radius - 50);

    var labelArc = d3.arc()
        .outerRadius(radius)
        .innerRadius(radius);

    //Pie Generator
    var pie = d3.pie()
        .sort(null)
        .value(function (d) {
            return d.prob
        });

    //Appending SVG
    var svg = d3.select("#donut-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + width / 2 + ", " + height / 2 + ")"); //Set the group object to the middlepoint

    //Appending g elements and arcs
    var g = svg.selectAll(".arc")
        .data(pie(data))
        .enter()
        .append("g")
        .attr("class", "arc");

    //Append path of the arc == Curve of the arc
    g.append("path")
        .attr("d", arc)
        .style("fill", function (d) {
            return color(d.data.class)
        })
        .transition() //Adding the transition
        .ease(d3.easeLinear)
        .duration(1500)
        .attrTween("d", transitionPie);

    //append labels
    g.append("text")
        .attr("transform", function (d) {
            return "translate(" + labelArc.centroid(d) + ")"
        })
        .attr("dy", "0px")
        .attr("dx", "0px")
        .text(function (d) {
            return "" + d.data.prob + "%"
        });

    function transitionPie(_arc) {
        _arc.innerRadius = 0;
        var i = d3.interpolate({startAngle: 0, endAngle: 0}, _arc);
        return function (t) {
            return arc(i(t))
        }
    }


}

