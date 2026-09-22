# Complementarity Case Studies

## RF_correct__VLM_correct
### full_rq0_seed42__rq0_0145__rq0_0153
- snippets: `rq0_0145` vs `rq0_0153`
- difficulty: `hard`
- human z: -1.5143 vs -1.3132; gold: `rq0_0153`
- RF: score_a=0.3039, score_b=0.3041, margin=-0.0002, pred=`rq0_0153`, correct=True
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0153`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0145.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0153.png`

Code A excerpt:
```java
	
	public long
	getInterval();
	
	public long
	getMinInterval();
	
	public int
	getTimeUntilNextUpdate();

```
Code B excerpt:
```java
	 */
	public void setGadgetKey(String gadgetKey);

	/**
	 * Returns the service name of this o auth token.
	 *
	 * @return the service name of this o auth token
	 */
	@AutoEscape
	public String getServiceName();

```

## RF_correct__VLM_wrong
### full_rq0_seed42__rq0_0227__rq0_0242
- snippets: `rq0_0227` vs `rq0_0242`
- difficulty: `medium`
- human z: 1.0832 vs 0.5387; gold: `rq0_0227`
- RF: score_a=0.0678, score_b=0.0640, margin=0.0038, pred=`rq0_0227`, correct=True
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0242`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0227.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0242.png`

Code A excerpt:
```java
/**
     * @return a list of all the types
     */
    public static Vector getTypes() {
        if (types == null) {
            types = new Vector();
            types.addElement(new KnowledgeTypeNode(Critic.KT_DESIGNERS));
            types.addElement(new KnowledgeTypeNode(Critic.KT_CORRECTNESS));
            types.addElement(new KnowledgeTypeNode(Critic.KT_COMPLETENESS));
            types.addElement(new KnowledgeTypeNode(Critic.KT_CONSISTENCY));
            types.addElement(new KnowledgeTypeNode(Critic.KT_SYNTAX));
            types.addElement(new KnowledgeTypeNode(Critic.KT_SEMANTICS));
            types.addElement(new KnowledgeTypeNode(Critic.KT_OPTIMIZATION));
            types.addElement(new KnowledgeTypeNode(Critic.KT_PRESENTATION));
            types.addElement(new KnowledgeTypeNode(Critic.KT_ORGANIZATIONAL));
            types.addElement(new KnowledgeTypeNode(Critic.KT_EXPERIENCIAL));
            types.addElement(new KnowledgeTypeNode(Critic.KT_TOOL));
        }
        return types;
    }
```
Code B excerpt:
```java
public String toString() {
		return new StringBuilder()
				.append("QueryStatistics")
				.append("[cacheHitCount=").append(this.cacheHitCount)
				.append(",cacheMissCount=").append(this.cacheMissCount)
				.append(",cachePutCount=").append(this.cachePutCount)
				.append(",executionCount=").append(this.executionCount)
				.append(",executionRowCount=").append(this.executionRowCount)
				.append(",executionAvgTime=").append(this.getExecutionAvgTime())
				.append(",executionMaxTime=").append(this.executionMaxTime)
				.append(",executionMinTime=").append(this.executionMinTime)
				.append(']')
				.toString();
	}
```

## RF_correct__VLM_invalid
### full_rq0_seed42__rq0_0096__rq0_0198
- snippets: `rq0_0096` vs `rq0_0198`
- difficulty: `hard`
- human z: -0.4425 vs -0.7319; gold: `rq0_0096`
- RF: score_a=-0.1013, score_b=-0.1034, margin=0.0021, pred=`rq0_0096`, correct=True
- VLM: AB=`B`, BA=`B`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0096.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0198.png`

Code A excerpt:
```java
			Description description= Description.createSuiteDescription(name);
			int n= ts.testCount();
			for (int i= 0; i < n; i++)
				description.addChild(makeDescription(ts.testAt(i)));

```
Code B excerpt:
```java
private void myDoubleClick(Object src) {
	Object sel = null;
	Diagram d = null;
	if (src == resultsTable) {
	    int row = resultsTable.getSelectionModel().getMinSelectionIndex();
	    if (row < 0) {
                return;
            }
	    sel = results.elementAt(row);
	    d = (Diagram) diagrams.elementAt(row);
	} else if (src == relatedTable) {
	    int row = relatedTable.getSelectionModel().getMinSelectionIndex();
	    if (row < 0) {
                return;
            }
	    numJumpToRelated++;
	    sel = related.elementAt(row);
	}

	if (d != null) {
            LOG.debug("go " + sel + " in " + d.getName());
            TargetManager.getInstance().setTarget(d);
        }
	TargetManager.getInstance().setTarget(sel);
    }
```

## RF_wrong__VLM_correct
### full_rq0_seed42__rq0_0021__rq0_0208
- snippets: `rq0_0021` vs `rq0_0208`
- difficulty: `hard`
- human z: 1.0323 vs 1.4462; gold: `rq0_0208`
- RF: score_a=-0.0492, score_b=-0.0511, margin=0.0019, pred=`rq0_0021`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0208`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0021.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0208.png`

Code A excerpt:
```java
	protected void printFailures(Result result) {
		if (result.getFailureCount() == 0)
			return;
		if (result.getFailureCount() == 1)
			getWriter().println("There was " + result.getFailureCount() + " failure:");
		else
			getWriter().println("There were " + result.getFailureCount() + " failures:");

```
Code B excerpt:
```java
public String extractConstraintName(SQLException sqle) {
			try {
				final int sqlState = Integer.valueOf( JdbcExceptionHelper.extractSqlState( sqle ) );
				switch (sqlState) {
					// CHECK VIOLATION
					case 23514: return extractUsingTemplate( "violates check constraint \"","\"", sqle.getMessage() );
					// UNIQUE VIOLATION
					case 23505: return extractUsingTemplate( "violates unique constraint \"","\"", sqle.getMessage() );
					// FOREIGN KEY VIOLATION
					case 23503: return extractUsingTemplate( "violates foreign key constraint \"","\"", sqle.getMessage() );
					// NOT NULL VIOLATION
					case 23502: return extractUsingTemplate( "null value in column \"","\" violates not-null constraint", sqle.getMessage() );
					// TODO: RESTRICT VIOLATION
					case 23001: return null;
					// ALL OTHER
					default: return null;
				}
			}
			catch (NumberFormatException nfe) {
				return null;
			}
		}
```

### full_rq0_seed42__rq0_0095__rq0_0310
- snippets: `rq0_0095` vs `rq0_0310`
- difficulty: `easy`
- human z: -0.6006 vs -1.8209; gold: `rq0_0095`
- RF: score_a=-0.6737, score_b=-0.6713, margin=-0.0024, pred=`rq0_0310`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0095`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0095.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0310.png`

Code A excerpt:
```java
    public ActionMenu getButtonAction() {
        AbstractAction action = new AbstractAction() {

            public void actionPerformed(ActionEvent evt) {
                showDialog();
            }
        };
        action.putValue(Action.NAME, mLocalizer.msg("CapturePlugin", "Capture Plugin"));
        action.putValue(Action.SMALL_ICON, createImageIcon("mimetypes", "video-x-generic", 16));

```
Code B excerpt:
```java
public AbstractRowReader(ReaderCollector readerCollector) {
		this.entityReferenceInitializers = readerCollector.getEntityReferenceInitializers() != null
				? new ArrayList<EntityReferenceInitializer>( readerCollector.getEntityReferenceInitializers() )
				: Collections.<EntityReferenceInitializer>emptyList();
		this.arrayReferenceInitializers = readerCollector.getArrayReferenceInitializers() != null
				? new ArrayList<CollectionReferenceInitializer>( readerCollector.getArrayReferenceInitializers() )
				: Collections.<CollectionReferenceInitializer>emptyList();
		this.collectionReferenceInitializers = readerCollector.getNonArrayCollectionReferenceInitializers() != null
				? new ArrayList<CollectionReferenceInitializer>( readerCollector.getNonArrayCollectionReferenceInitializers() )
				: Collections.<CollectionReferenceInitializer>emptyList();
	}
```

### full_rq0_seed42__rq0_0096__rq0_0207
- snippets: `rq0_0096` vs `rq0_0207`
- difficulty: `medium`
- human z: -0.4425 vs 0.5387; gold: `rq0_0207`
- RF: score_a=-0.1013, score_b=-0.1080, margin=0.0067, pred=`rq0_0096`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0207`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0096.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0207.png`

Code A excerpt:
```java
			Description description= Description.createSuiteDescription(name);
			int n= ts.testCount();
			for (int i= 0; i < n; i++)
				description.addChild(makeDescription(ts.testAt(i)));

```
Code B excerpt:
```java
/**
	 * Constructs a SybaseASE157Dialect
	 */
	public SybaseASE157Dialect() {
		super();

		registerFunction( "create_locator", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "create_locator(?1, ?2)" ) );
		registerFunction( "locator_literal", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "locator_literal(?1, ?2)" ) );
		registerFunction( "locator_valid", new SQLFunctionTemplate( StandardBasicTypes.BOOLEAN, "locator_valid(?1)" ) );
		registerFunction( "return_lob", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "return_lob(?1, ?2)" ) );
		registerFunction( "setdata", new SQLFunctionTemplate( StandardBasicTypes.BOOLEAN, "setdata(?1, ?2, ?3)" ) );
		registerFunction( "charindex", new SQLFunctionTemplate( StandardBasicTypes.INTEGER, "charindex(?1, ?2, ?3)" ) );
	}
```

### full_rq0_seed42__rq0_0124__rq0_0288
- snippets: `rq0_0124` vs `rq0_0288`
- difficulty: `medium`
- human z: 0.8283 vs 1.4462; gold: `rq0_0288`
- RF: score_a=-0.0776, score_b=-0.0857, margin=0.0081, pred=`rq0_0124`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0288`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0124.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0288.png`

Code A excerpt:
```java
				WorkflowConstants.CONTEXT_ENTRY_CLASS_NAME));

		if (workflowContext.containsKey(
				WorkflowConstants.CONTEXT_ENTRY_CLASS_PK)) {

			kaleoInstanceToken.setClassPK(
				GetterUtil.getLong(
					(String)workflowContext.get(
						WorkflowConstants.CONTEXT_ENTRY_CLASS_PK)));
		}

```
Code B excerpt:
```java
@Override
	protected XMLEvent internalNextEvent() throws XMLStreamException {
		//If there is an iterator to read from reset was called, use the iterator
		//until it runs out of events.
		if (this.bufferReader != null) {
			final XMLEvent event = this.bufferReader.next();

			//If nothing left in the iterator, remove the reference and fall through to direct reading
			if (!this.bufferReader.hasNext()) {
				this.bufferReader = null;
			}

			return event;
		}

		//Get the next event from the underlying reader
		final XMLEvent event = this.getParent().nextEvent();

		//if buffering add the event
		if (this.eventLimit != 0) {
			this.eventBuffer.offer(event);

			//If limited buffer size and buffer is too big trim the buffer.
			if (this.eventLimit > 0 && this.eventBuffer.size() > this.eventLimit) {
				this.eventBuffer.poll();
			}
		}

		return event;
	}
```

### full_rq0_seed42__rq0_0177__rq0_0228
- snippets: `rq0_0177` vs `rq0_0228`
- difficulty: `easy`
- human z: 0.9977 vs -0.5504; gold: `rq0_0177`
- RF: score_a=-0.0785, score_b=-0.0690, margin=-0.0095, pred=`rq0_0228`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0177`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0177.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0228.png`

Code A excerpt:
```java
/*
 * Copyright (C) 2007 Rob Manning
 * manningr@users.sourceforge.net
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
package net.sourceforge.squirrel_sql.plugins.dbdiff;

import net.sourceforge.squirrel_sql.client.session.ISession;
import net.sourceforge.squirrel_sql.fw.sql.IDatabaseObjectInfo;

/**
 * This is implemented in order to pass needed info along to diff executor.
 */
public interface SessionInfoProvider
{

	public void setSourceSession(ISession session);

	public ISession getSourceSession();

	public IDatabaseObjectInfo[] getSourceSelectedDatabaseObjects();

	public IDatabaseObjectInfo[] getDestSelectedDatabaseObjects();

	public void setDestSelectedDatabaseObjects(IDatabaseObjectInfo[] infos);

	public void setSourceSelectedDatabaseObjects(IDatabaseObjectInfo[] infos);

	public void setDestSession(ISession session);

	public ISession getDestSession();

	/**
	 * @return the scriptFileManager
	 */
	public IScriptFileManager getScriptFileManager();

```
Code B excerpt:
```java
public void toDoItemsRemoved(ToDoListEvent tde) {
	LOG.debug("toDoItemRemoved");
        Vector items = tde.getToDoItems();
        int nItems = items.size();
        
	ToDoList list = Designer.theDesigner().getToDoList(); //source?
	Object[] path = new Object[2];
	path[0] = Designer.theDesigner().getToDoList();


	Enumeration elems = list.getPosters().elements();
 	while (elems.hasMoreElements()) {
	    Poster p = (Poster) elems.nextElement();
            boolean anyInPoster = false;
            for (int i = 0; i < nItems; i++) {
                ToDoItem item = (ToDoItem) items.elementAt(i);
                Poster post = item.getPoster();
                if (post == p) { 
                    anyInPoster = true;
                    break;
                }
            }
            if (!anyInPoster) { 
                continue;
            }
	    path[1] = p;
	    fireTreeStructureChanged(path);
	}
    }
```

### full_rq0_seed42__rq0_0287__rq0_0290
- snippets: `rq0_0287` vs `rq0_0290`
- difficulty: `medium`
- human z: 1.8092 vs 1.0832; gold: `rq0_0287`
- RF: score_a=0.4423, score_b=0.4526, margin=-0.0103, pred=`rq0_0290`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0287`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0287.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0290.png`

Code A excerpt:
```java
@Override
	public void release() {
		if ( reader == null ) {
			return;
		}
		try {
			reader.close();
		}
		catch (IOException ignore) {
		}
	}
```
Code B excerpt:
```java
public Point getClosestPoint(Point anotherPt) {
        Rectangle r = getBounds();
        int[] xs = {r.x + r.width / 2,
                    r.x + r.width,
                    r.x + r.width / 2,
                    r.x,
                    r.x + r.width / 2,
        };
        int[] ys = {r.y,
                    r.y + r.height / 2,
                    r.y + r.height,
                    r.y + r.height / 2,
                    r.y,
        };
        Point p =
            Geometry.ptClosestTo(
                xs,
                ys,
                5,
                anotherPt);
        return p;
    }
```

### full_rq0_seed42__rq0_0169__rq0_0267
- snippets: `rq0_0169` vs `rq0_0267`
- difficulty: `medium`
- human z: -0.0065 vs -0.5504; gold: `rq0_0169`
- RF: score_a=0.1005, score_b=0.1112, margin=-0.0107, pred=`rq0_0267`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0169`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0169.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0267.png`

Code A excerpt:
```java
			WebKeys.MOBILE_DEVICE_RULES_RULE_EDITOR_JSP, editorJSP);

		long ruleGroupId = BeanParamUtil.getLong(
			rule, renderRequest, "ruleGroupId");

		MDRRuleGroup ruleGroup = MDRRuleGroupServiceUtil.getRuleGroup(
			ruleGroupId);

		renderRequest.setAttribute(
			WebKeys.MOBILE_DEVICE_RULES_RULE_GROUP, ruleGroup);

		return mapping.findForward("portlet.mobile_device_rules.edit_rule");
	}

	@Override
	public void serveResource(
			ActionMapping mapping, ActionForm form, PortletConfig portletConfig,
			ResourceRequest resourceRequest, ResourceResponse resourceResponse)
		throws Exception {

		long ruleId = ParamUtil.getLong(resourceRequest, "ruleId");

		if (ruleId > 0) {
			MDRRule rule = MDRRuleServiceUtil.fetchRule(ruleId);

			resourceRequest.setAttribute(
				WebKeys.MOBILE_DEVICE_RULES_RULE, rule);
		}

		String type = ParamUtil.getString(resourceRequest, "type");

```
Code B excerpt:
```java
public final void caseSList() throws RecognitionException, TokenStreamException {
		
		
		{
		_loop119:
		do {
			if ((_tokenSet_6.member(LA(1)))) {
				statement();
			}
			else {
				break _loop119;
			}
			
		} while (true);
		}
	}
```

### full_rq0_seed42__rq0_0109__rq0_0268
- snippets: `rq0_0109` vs `rq0_0268`
- difficulty: `hard`
- human z: 0.1577 vs -0.1874; gold: `rq0_0109`
- RF: score_a=0.1750, score_b=0.1864, margin=-0.0114, pred=`rq0_0268`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0109`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0109.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0268.png`

Code A excerpt:
```java
		return (Address)message.get(_ADDRESS);
	}

	public static ClusterLink getClusterLink() {
		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

			return null;
		}

		return _clusterLink;
	}

	public static List<Address> getLocalTransportAddresses() {
		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

			return Collections.emptyList();
		}

		return _clusterLink.getLocalTransportAddresses();
	}

	public static List<Address> getTransportAddresses(Priority priority) {
		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

			return Collections.emptyList();
		}

		return _clusterLink.getTransportAddresses(priority);
	}

	public static boolean isForwardMessage(Message message) {
		return message.getBoolean(CLUSTER_FORWARD_MESSAGE);
	}

	public static void sendMulticastMessage(
		Message message, Priority priority) {

		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

```
Code B excerpt:
```java
public void write(BufferedReader reader,
                      BufferedWriter writer,
                      Stack parseStateStack) throws IOException {
        ParseState parseState = (ParseState) parseStateStack.peek();
        Object mInterface = /*(MInterface)*/ parseState.newClassifier(name);

	if (mInterface != null) {
	    parseStateStack.push(new ParseState(mInterface));
	    StringBuffer sbText =
		GeneratorJava.getInstance().generateClassifierStart(mInterface);
	    if (sbText != null) {
		writer.write (sbText.toString());
	    }
            // dispose code piece in reader
            ffCodePiece(reader, null);
        } else {
            // not in model, so write the original code
            ffCodePiece(reader, writer);
        }
    }
```

### full_rq0_seed42__rq0_0288__rq0_0294
- snippets: `rq0_0288` vs `rq0_0294`
- difficulty: `easy`
- human z: 1.4462 vs -1.8209; gold: `rq0_0288`
- RF: score_a=-0.0857, score_b=-0.0734, margin=-0.0123, pred=`rq0_0294`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0288`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0288.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0294.png`

Code A excerpt:
```java
@Override
	protected XMLEvent internalNextEvent() throws XMLStreamException {
		//If there is an iterator to read from reset was called, use the iterator
		//until it runs out of events.
		if (this.bufferReader != null) {
			final XMLEvent event = this.bufferReader.next();

			//If nothing left in the iterator, remove the reference and fall through to direct reading
			if (!this.bufferReader.hasNext()) {
				this.bufferReader = null;
			}

			return event;
		}

		//Get the next event from the underlying reader
		final XMLEvent event = this.getParent().nextEvent();

		//if buffering add the event
		if (this.eventLimit != 0) {
			this.eventBuffer.offer(event);

			//If limited buffer size and buffer is too big trim the buffer.
			if (this.eventLimit > 0 && this.eventBuffer.size() > this.eventLimit) {
				this.eventBuffer.poll();
			}
		}

		return event;
	}
```
Code B excerpt:
```java
public void buildModel() {
        if (getTarget() != null) {
            Object target = getTarget();
            Object kind = Model.getFacade().getAggregation(target);
            if (kind == null
                    || kind.equals(
                            Model.getAggregationKind().getNone())) {
                setSelected(ActionSetAssociationEndAggregation.NONE_COMMAND);
            } else {
		if (kind.equals(
		        Model.getAggregationKind().getAggregate())) {
		    setSelected(ActionSetAssociationEndAggregation
		            .AGGREGATE_COMMAND);
		} else {
		    if (kind.equals(
		            Model.getAggregationKind()
		            	.getComposite())) {
			setSelected(ActionSetAssociationEndAggregation
			        .COMPOSITE_COMMAND);
		    } else {
		        setSelected(ActionSetAssociationEndAggregation

			        .NONE_COMMAND);
		    }
		}
            }
        }
    }
```

### full_rq0_seed42__rq0_0063__rq0_0179
- snippets: `rq0_0063` vs `rq0_0179`
- difficulty: `hard`
- human z: -0.1133 vs -0.6062; gold: `rq0_0063`
- RF: score_a=0.0752, score_b=0.0875, margin=-0.0123, pred=`rq0_0179`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0063`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0063.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0179.png`

Code A excerpt:
```java

        String[] texts = new String[messages.length];
        ImageIcon[] images = new ImageIcon[messages.length];
        for (int i = 0; i < messages.length; i++) {
            String ID = messages[i].getMessageID();

```
Code B excerpt:
```java
				sql = _SQL_SELECT_SCPRODUCTVERSION.concat(SCProductVersionModelImpl.ORDER_BY_JPQL);
			}

			Session session = null;

			try {
				session = openSession();

				Query q = session.createQuery(sql);

				if (orderByComparator == null) {
					list = (List<SCProductVersion>)QueryUtil.list(q,
							getDialect(), start, end, false);

					Collections.sort(list);
				}
				else {
					list = (List<SCProductVersion>)QueryUtil.list(q,
							getDialect(), start, end);
				}
			}
			catch (Exception e) {
				throw processException(e);
			}
			finally {
				if (list == null) {
					FinderCacheUtil.removeResult(finderPath, finderArgs);
				}
				else {
					cacheResult(list);

```

## RF_wrong__VLM_wrong
### full_rq0_seed42__rq0_0069__rq0_0133
- snippets: `rq0_0069` vs `rq0_0133`
- difficulty: `hard`
- human z: 0.6504 vs 0.4328; gold: `rq0_0069`
- RF: score_a=0.4297, score_b=0.4318, margin=-0.0021, pred=`rq0_0133`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0133`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0069.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0133.png`

Code A excerpt:
```java
	/**
		Translate bsh.Modifiers into ASM modifier bitflags.
	*/
	static int getASMModifiers( Modifiers modifiers ) 
	{
		int mods = 0;
		if ( modifiers == null )
			return mods;

		if ( modifiers.hasModifier("public") )
			mods += ACC_PUBLIC;

```
Code B excerpt:
```java

		Date createDate = getCreateDate();

		if (createDate != null) {
			passwordPolicyCacheModel.createDate = createDate.getTime();
		}
		else {
			passwordPolicyCacheModel.createDate = Long.MIN_VALUE;
		}

		Date modifiedDate = getModifiedDate();

		if (modifiedDate != null) {
			passwordPolicyCacheModel.modifiedDate = modifiedDate.getTime();
		}
		else {
			passwordPolicyCacheModel.modifiedDate = Long.MIN_VALUE;
		}

		passwordPolicyCacheModel.defaultPolicy = getDefaultPolicy();

		passwordPolicyCacheModel.name = getName();

		String name = passwordPolicyCacheModel.name;

		if ((name != null) && (name.length() == 0)) {
			passwordPolicyCacheModel.name = null;
		}

		passwordPolicyCacheModel.description = getDescription();

```

## RF_wrong__VLM_invalid
### full_rq0_seed42__rq0_0021__rq0_0308
- snippets: `rq0_0021` vs `rq0_0308`
- difficulty: `easy`
- human z: 1.0323 vs -0.9134; gold: `rq0_0021`
- RF: score_a=-0.0492, score_b=-0.0488, margin=-0.0003, pred=`rq0_0308`, correct=False
- VLM: AB=`B`, BA=`B`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0021.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0308.png`

Code A excerpt:
```java
	protected void printFailures(Result result) {
		if (result.getFailureCount() == 0)
			return;
		if (result.getFailureCount() == 1)
			getWriter().println("There was " + result.getFailureCount() + " failure:");
		else
			getWriter().println("There were " + result.getFailureCount() + " failures:");

```
Code B excerpt:
```java
public static <T> JaxbRoot<T> unmarshallXml(String fileName, String schemaName, Class<T> clazz, ClassLoaderService classLoaderService)
            throws JAXBException {
        Schema schema = getMappingSchema( schemaName, classLoaderService );
        InputStream in = classLoaderService.locateResourceStream( fileName );
        JAXBContext jc = JAXBContext.newInstance( clazz );
        Unmarshaller unmarshaller = jc.createUnmarshaller();
        unmarshaller.setSchema( schema );
        StreamSource stream = new StreamSource( in );
        JAXBElement<T> elem = unmarshaller.unmarshal( stream, clazz );
        Origin origin = new Origin( null, fileName );
        return new JaxbRoot<T>( elem.getValue(), origin );
    }
```

