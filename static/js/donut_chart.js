function donutChart(data) {
    data = data["forecast"]; //Place in dict where final prediction values are stored
    // console.log(data);
    let _sum = 0;
    data.forEach(function (d) {
        _sum += parseFloat(d["prob"])
    });
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
        width = $("#donut-chart").width() - margin.right - margin.left,
        height = 400 - margin.top - margin.bottom,
        radius = width / 3.0;

    var color = d3.scaleOrdinal()
        .range(["#b4126b", "#007581", "#055E67", "#C0AD26"]); //Color range
    // Arc Generator produces the arcs for the pie generator
    var arc = d3.arc()
        .outerRadius(radius - 50)
        .innerRadius(radius - 100);

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
        .attr("transform", "translate(" + width / 2.8 + ", " + height / 2 + ")"); //Set the group object to the middlepoint (close)

    //Appending g elements and arcs
    var g = svg.selectAll(".arc")
        .data(pie(data))
        .enter()
        .append("g")
        .attr("class", "arc")
        .attr("id", function (d) {
            return d.data.class
        });

    //Legend
    var legend = svg.selectAll(".legend")
        .data(pie(data))
        .enter()
        .append("g")
        .attr("class", "legend")
        .style("cursor", "pointer")
        .attr("legend-id", function (d) {
            return d.data.class
        })
        .attr("transform", function (d, i) {
            return "translate(30," + (-70 * i) + ")"
        });
    // Legend Entry
    var leg = legend.append("rect");

    //Tooltip for arcs
    var tooltip = svg.append("g")
        .attr("class", "tooltip")
        .style("cursor", "pointer");

    //Build Legend
    //Change x if you want to have the rectangles on the other side
    leg.attr("x", width / 2)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", function (d) {
            return color(d.data.class)
        });

    legend.append("text")
        .attr("x", (width / 2) - 5)
        .attr("dy", ".8em")
        .style("text-anchor", "end")
        .text(function (d) {
            return d.data.class
        });

    //Build Tooltip
    tooltip.append("text")
        .attr("x", 15)
        .attr("dy", "1.2em")
        .style("font-size", "1.6em")
        .style("pointer-events", "none");

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

    // Mouseover effects
    //Legend
    legend.on("mouseover", function (d) {

        $("#" + d.data.class ).find("path").css("transform", "scale(1.15)");

    })
        .on("mouseout", function (d) {
            $("#" + d.data.class).find("path").css("transform", "scale(1.0)");
        });

    //Tooltip
    g.on("mouseover", function () {
            tooltip.style("display", "block")
        }
    )
        .on("mouseout", function () {
            tooltip.style("display", "none")
        })
        .on("mousemove", function (d) {
            var xPos = d3.mouse(this)[0] - 15;
            var yPos = d3.mouse(this)[1] - 15;
            tooltip.attr("transform", "translate(" + xPos + ", " + yPos + ")");
            tooltip.select("text").text((d.data.prob/_sum * 100).toFixed(2) + "%");
        });


    function transitionPie(_arc) {
        _arc.innerRadius = 0;
        var i = d3.interpolate({startAngle: 0, endAngle: 0}, _arc);
        return function (t) {
            return arc(i(t))
        }
    }


}

