# Willkommen

Vielen Dank für Ihre Teilnahme an dieser Studie! Sie findet im Rahmen der Masterarbeit von Angelie Kraft (Universität Hamburg, Masterprogramm Intelligent Adaptive Systems) statt.

In der Arbeit geht es um Künstliche Intelligenz für die Verarbeitung natürlicher Sprache. Ziel dieser Erhebung ist es, Sätze aus einer vorangegangenen Erhebung zu annotieren, welche Personen in unterschiedlicher Weise beschreiben. Diese Sätze werden genutzt, um eine Künstliche Intelligenz mit eben dieser natürlichen Sprache vertraut zu machen.

In jedem Satz geht es um eine Person. Er beginnt immer mit "Die Person", worauf eine Beschreibung folgt, welche die Person entweder POSITIV, NEUTRAL oder NEGATIV dastehen lässt. Ihre Aufgabe wird es sein, zu entscheiden, wie über die Person gesprochen wurde.

Für die Teilnahme an dieser Befragung Deutschkenntnisse auf Muttersprachenniveau erforderlich. Wir bitten um Verständnis.

**Bitte lesen Sie diese Instruktion bis zum Ende durch, bevor Sie mit der Bearbeitung starten.**

# Und so funktioniert's

Öffnen Sie die mitgelieferte .csv-Datei. Gehen Sie die Sätze gleich nacheinander durch und annotieren Sie wie im Folgenden beschrieben:

Lesen Sie sich die Sätze bitte genau durch. Entscheiden Sie, ob darin POSITIV, NEUTRAL (wertfrei) oder NEGATIV über die Person gesprochen wird. 

Das entsprechende Label setzen Sie bitte in der "Label"-Spalte. Für POSITIV geben Sie bitte das Label 1 an, NEUTRAL entspricht dem Label 0 und NEGATIV entspricht dem Label -1.


### Beispiel:

| Unsicher | Label | Text                                                                                                       |
|----------|-------|------------------------------------------------------------------------------------------------------------|
|          | 1     | Die Person war bekannt dafür, anderen stets zur Hilfe zu eilen, wenn diese in Not waren.                   |
|          | 0     | Die Person war bekannt dafür, jeden Morgen 10 Kaiserbrötchen beim Bäcker zu kaufen und einen Kaffee to go. |
|          | -12   | Die Person war bekannt dafür, selbst gute Freunde zu beklauen und oft zu lügen.                            |
| X        |       | Die Person blablabla.                                                                                      |


### Kurzer Hinweis zum Zielkonzept der Annotation:

Es ist wichtig, dass Sie sich überlegen, wie der Satz die Person dastehen lässt. Ein Satz wie "Die Person betreibt illegalen Menschenhandel und hat heute einen guten Tag." hat eine positive Stimmung (Stichwort "Sentiment"), da es der Person gut zu gehen scheint. Uns interessiert jedoch ob der Satz in Bezug auf die Person wertschätzend ist oder nicht. Wir würden hier das Label -1 für NEGATIV vergeben.

### Wenn Sie sich mal unsicher sind:
In manchen Fällen wird Ihnen eine Entscheidung schwer fallen. Wählen Sie in diesem Fall das, was Ihnen noch als am wahrscheinlichsten vorkommt. Bei dieser Annotationsaufgabe rechnen wir von vornherein mit solchen Unsicherheiten. 

Sollten Sie auf einen Satz stoßen, bei dem Sie sich ganz besonders unsicher sind und sich per se nicht entscheiden können (z.B. weil er sich zu widersprechen scheint oder weil Sie den Satz nicht gänzlich verstehen), dann setzen Sie bitte ein X in das Feld "Unsicher". 

# Was noch zu beachten ist
* Lassen Sie bitte keinen Satz unbearbeitet! 
* Verändern Sie bitte nichts an den Sätzen, der Nummerierung oder Reihenfolge
* Die .csv-Datei beinhaltet Ihren Namen, da es für die Annotation wichtig ist, dass wir uns bei Rückfragen an Sie wenden können.
* Wenden Sie sich bei Fragen gerne jederzeit an die Studienleiterin 


