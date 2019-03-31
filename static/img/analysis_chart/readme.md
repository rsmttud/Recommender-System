# Usage 

The directory contains all the svg files which are used to create an example plot in the result page. 
Import you svg here if you want to add another class description. If you want that the names of entities are added to 
your chart, use group the corresponding elements in a group with the id _custom_ and include the corresponding SVGs with text field according to the ids: _entity-placeholder-1_ or _enitiy-placeholder-2_ as shown below:

Please note the described procedure only works for 2 Entities, which are shown in the figure.
```
<g id="custom">
    <g transform="matrix(1,0,0,1,206.06,255.678)">
        <text id = "entity-placeholder-1" x="0px" y="8.596px" style="font-family:'ArialMT', 'Arial', sans-serif;font-size:12px;">Entity Placeholder I</text>
    </g>
    <g transform="matrix(0.0158606,-0.999874,0.999874,0.0158606,26.5052,154.901)">
        <text id = "entity-placeholder-2" x="0px" y="8.596px" style="font-family:'ArialMT', 'Arial', sans-serif;font-size:12px;">Entity Placeholder II</text>
    </g>
</g>
``` 