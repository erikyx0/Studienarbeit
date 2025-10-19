# Vergleich von Reaktornetzwerkmodellen für die nichtkatalytische Partialoxidation von Erdgas

## Übersicht
Dieses Repository dokumentiert die im Rahmen meiner Studienarbeit an der  
**TU Bergakademie Freiberg (2025)** durchgeführte Untersuchung zur  
**nichtkatalytischen Partialoxidation (POx) von Erdgas**.  
Ziel der Arbeit ist der Aufbau, Vergleich und die Bewertung reduzierter Reaktormodelle  
(*Reduced-Order Models, ROMs*) zur Abbildung der experimentellen Versuchsdaten  
des Projekts **SCOORE (Synthesis gas from recycling of CO₂)**.

---

## Zielsetzung
- Entwicklung verschiedener reduzierter Reaktornetzwerke auf Basis idealisierter Reaktoren  
  (Perfectly Stirred Reactor, Plug Flow Reactor)  
- Untersuchung des Einflusses der **Netzwerkkomplexität** auf Modellgüte und Rechenzeit  
- Vergleich unterschiedlicher **Reaktionsmechanismen** (GRI-Mech, Aramco, NUIG, CRECK, ATR)  
- Bewertung der Modellgüte anhand von **experimentellen Daten der Versuchskampagne GASPOX215**  
- Durchführung einer **Parameterstudie** zum Einfluss des CO₂-Feedstroms auf Temperatur und Produktgaszusammensetzung  

---

## Methodik
- Aufbau mehrerer Reaktornetzwerk-Topologien (PSR/PFR-Kombinationen)  
- Simulationen mit **ANSYS Chemkin-Pro** und **Cantera (Python)**  
- Vergleich der Modellresultate hinsichtlich Temperaturverlauf, Stoffumwandlung und H₂/CO-Verhältnis  
- Bewertung der Modelle mittels **mittlerem quadratischem Fehler (MSE)** und visueller Gegenüberstellung zu Messwerten  
- Schrittweise Erweiterung der Netzwerke zur Abbildung von **Bypass- und Rezirkulationszonen**

---

## Zentrale Ergebnisse
- Alle betrachteten Reaktionsmechanismen liefern vergleichbare Resultate für die Hauptkomponenten (H₂, CO, CH₄, CO₂).  
- Der **Aramco-Mechanismus** zeigt das beste Verhältnis aus Genauigkeit und Rechenaufwand.  
- Mit steigender CO₂-Zugabe sinkt die Reaktionstemperatur und das H₂/CO-Verhältnis.  
- Erweiterte Reaktornetzwerke (mit Rezirkulation) erhöhen die Modellgüte signifikant bei moderatem Mehraufwand.  
- Das beste Verhältnis zwischen Genauigkeit und Komplexität wird bei mittlerer Netzwerkgröße erreicht.

---

## Bedeutung
Die Ergebnisse demonstrieren, dass ROMs mit wenigen idealisierten Reaktoren  
eine präzise Abbildung der POx-Prozesse ermöglichen und somit eine Grundlage  
für Prozessoptimierung, Mechanismenvergleich und Echtzeitanwendungen bieten.  
Das Projekt trägt zum besseren Verständnis der CO₂-Rückführung in Hochtemperaturprozessen bei  
und unterstützt die Entwicklung effizienter Synthesegasverfahren.

---

## Kontext
Die Arbeit entstand im Rahmen des **Verbundprojekts SCOORE (BASF / TU Bergakademie Freiberg)**  
und basiert auf experimentellen Daten aus der **Versuchsanlage in Leuna (GASPOX215)**.  
Betreuer: Prof. Dr. Andreas Richter, Gabriel Gonzales Ortiz.

---

## Zitation
> Domogalla, Erik (2025): *Vergleich von Reaktornetzwerkmodellen für die nichtkatalytische Partialoxidation von Erdgas*.  
> Studienarbeit, Technische Universität Bergakademie Freiberg.

