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
### full_rq0_seed42__rq0_0009__rq0_0094
- snippets: `rq0_0009` vs `rq0_0094`
- difficulty: `medium`
- human z: 0.0974 vs 0.8348; gold: `rq0_0094`
- RF: score_a=0.4553, score_b=0.4560, margin=-0.0006, pred=`rq0_0094`, correct=True
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0009`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0009.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0094.png`

Code A excerpt:
```java
    private void moveUnit(KeyEvent e) {
        if (!parent.isMapboardActionsEnabled()) {
            return;
        }
        
        switch (e.getKeyCode()) {
        case KeyEvent.VK_ESCAPE:
            // main menu
            break;
        case KeyEvent.VK_NUMPAD1:
        case KeyEvent.VK_END:
            inGameController.moveActiveUnit(Map.SW);

```
Code B excerpt:
```java
    /**
     * Applies this action.
     * 
     * @param e The <code>ActionEvent</code>.
     */
    public void actionPerformed(ActionEvent e) {
        final Game game = freeColClient.getGame();
        final Map map = game.getMap();

        Parameters p = showParametersDialog();

```

## RF_correct__VLM_invalid
### full_rq0_seed42__rq0_0159__rq0_0219
- snippets: `rq0_0159` vs `rq0_0219`
- difficulty: `hard`
- human z: 0.3834 vs 0.1757; gold: `rq0_0159`
- RF: score_a=-0.2449, score_b=-0.2457, margin=0.0008, pred=`rq0_0159`, correct=True
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0159.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0219.png`

Code A excerpt:
```java
import com.liferay.portal.model.User;
import com.liferay.portal.security.permission.ActionKeys;
import com.liferay.portal.service.base.EmailAddressServiceBaseImpl;
import com.liferay.portal.service.permission.CommonPermissionUtil;

import java.util.List;

/**
 * @author Brian Wing Shun Chan
 * @author Alexander Chow
 */
public class EmailAddressServiceImpl extends EmailAddressServiceBaseImpl {

	public EmailAddress addEmailAddress(
			String className, long classPK, String address, int typeId,
			boolean primary)
		throws PortalException, SystemException {

		CommonPermissionUtil.check(
			getPermissionChecker(), className, classPK, ActionKeys.UPDATE);

		return emailAddressLocalService.addEmailAddress(
			getUserId(), className, classPK, address, typeId, primary);
	}

	public void deleteEmailAddress(long emailAddressId)
		throws PortalException, SystemException {

		EmailAddress emailAddress = emailAddressPersistence.findByPrimaryKey(
			emailAddressId);

		CommonPermissionUtil.check(
			getPermissionChecker(), emailAddress.getClassNameId(),
			emailAddress.getClassPK(), ActionKeys.UPDATE);

		emailAddressLocalService.deleteEmailAddress(emailAddressId);
	}

	public EmailAddress getEmailAddress(long emailAddressId)
		throws PortalException, SystemException {

		EmailAddress emailAddress = emailAddressPersistence.findByPrimaryKey(
			emailAddressId);

		CommonPermissionUtil.check(
			getPermissionChecker(), emailAddress.getClassNameId(),
			emailAddress.getClassPK(), ActionKeys.VIEW);

		return emailAddress;
	}

```
Code B excerpt:
```java
public int getSqlTypeCode(Mapping mapping) throws MappingException {
        org.hibernate.type.Type type = getValue().getType();
        try {
            int sqlTypeCode = type.sqlTypes( mapping )[getTypeIndex()];
            if ( getSqlTypeCode() != null && getSqlTypeCode() != sqlTypeCode ) {
                throw new MappingException( "SQLType code's does not match. mapped as " + sqlTypeCode + " but is " + getSqlTypeCode() );
            }
            return sqlTypeCode;
        }
        catch ( Exception e ) {
            throw new MappingException(
                    "Could not determine type for column " +
                            name +
                            " of type " +
                            type.getClass().getName() +
                            ": " +
                            e.getClass().getName(),
                    e
            );
        }
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

### full_rq0_seed42__rq0_0094__rq0_0290
- snippets: `rq0_0094` vs `rq0_0290`
- difficulty: `hard`
- human z: 0.8348 vs 1.0832; gold: `rq0_0290`
- RF: score_a=0.4560, score_b=0.4526, margin=0.0033, pred=`rq0_0094`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0290`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0094.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0290.png`

Code A excerpt:
```java
    /**
     * Applies this action.
     * 
     * @param e The <code>ActionEvent</code>.
     */
    public void actionPerformed(ActionEvent e) {
        final Game game = freeColClient.getGame();
        final Map map = game.getMap();

        Parameters p = showParametersDialog();

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

### full_rq0_seed42__rq0_0037__rq0_0300
- snippets: `rq0_0037` vs `rq0_0300`
- difficulty: `medium`
- human z: -0.5084 vs -0.0059; gold: `rq0_0300`
- RF: score_a=-0.3125, score_b=-0.3189, margin=0.0065, pred=`rq0_0037`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0300`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0037.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0300.png`

Code A excerpt:
```java
            row[1] = ns.getCatalogName(row[0]);
            row[2] = schema.equals(defschema) ? Boolean.TRUE
                                              : Boolean.FALSE;

            t.insertSys(row);

```
Code B excerpt:
```java
@Override
      public String call() throws Exception {
         try {
            if (isTrace)
               log.tracef("[%s] Wait for all executions paths to be ready to perform calls", title(warmup));
            barrier.await();

            long start = System.nanoTime();
            int runs = 0;
            if (isTrace)
               log.tracef("[%s] Start time: %d", title(warmup), start);

//            while (USE_TIME && PutFromLoadStressTestCase.this.run.get()) {
//               if (runs % 100000 == 0)
//                  log.infof("[%s] Query run # %d", title(warmup), runs);
//
////               Customer customer = query();
////               deleteCached(customer);

               queryItems();
//               deleteCachedItems();
//
//               runs++;
//            }
            long end = System.nanoTime();
            long duration = end - start;
            if (isTrace)
               log.tracef("[%s] End time: %d, duration: %d, runs: %d",
                     title(warmup), start, duration, runs);

            return opsPerMS(duration, runs);
         } finally {
            if (isTrace)
               log.tracef("[%s] Wait for all execution paths to finish", title(warmup));

            barrier.await();
         }
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

### full_rq0_seed42__rq0_0003__rq0_0045
- snippets: `rq0_0003` vs `rq0_0045`
- difficulty: `hard`
- human z: 1.1903 vs 0.9401; gold: `rq0_0003`
- RF: score_a=0.3389, score_b=0.3477, margin=-0.0088, pred=`rq0_0045`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0003`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0003.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0045.png`

Code A excerpt:
```java
    /**
     * Constructor, with a argument reference to the PUBLIC User Object which
     * is null if this is the SYS or PUBLIC user.
     *
     * The dependency upon a GranteeManager is undesirable.  Hopefully we
     * can get rid of this dependency with an IOC or Listener re-design.
     */
    Grantee(String name, Grantee inGrantee,
            GranteeManager man) throws HsqlException {

        rightsMap      = new IntValueHashMap();
        granteeName    = name;
        granteeManager = man;}

```
Code B excerpt:
```java
    /**
     * Read Settings
     * @param stream
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public void readData(ObjectInputStream stream) throws IOException, ClassNotFoundException {
      int version = stream.readInt();
      mNumber = stream.readInt();
      mName = stream.readUTF();

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

### full_rq0_seed42__rq0_0005__rq0_0153
- snippets: `rq0_0005` vs `rq0_0153`
- difficulty: `hard`
- human z: -1.7462 vs -1.3132; gold: `rq0_0153`
- RF: score_a=0.3189, score_b=0.3041, margin=0.0148, pred=`rq0_0005`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0153`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0005.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0153.png`

Code A excerpt:
```java
    xsp = jj_scanpos;
    if (jj_scan_token(100)) {
    jj_scanpos = xsp;
    if (jj_scan_token(101)) return true;

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

### full_rq0_seed42__rq0_0065__rq0_0156
- snippets: `rq0_0065` vs `rq0_0156`
- difficulty: `medium`
- human z: 1.1640 vs 0.2482; gold: `rq0_0065`
- RF: score_a=0.3887, score_b=0.4038, margin=-0.0151, pred=`rq0_0156`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0065`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0065.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0156.png`

Code A excerpt:
```java
  /**
   * Add one zero if neccessary
   * @param number
   * @return
   */
  private CharSequence addZero(int number) {
    StringBuilder builder = new StringBuilder();
    
    if (number < 10) {
      builder.append('0');
    }
    
    builder.append(Integer.toString(number));

```
Code B excerpt:
```java

		for (KaleoTimer model : models) {
			soapModels.add(toSoapModel(model));
		}

		return soapModels.toArray(new KaleoTimerSoap[soapModels.size()]);
	}

	public KaleoTimerSoap() {
	}

```

### full_rq0_seed42__rq0_0074__rq0_0205
- snippets: `rq0_0074` vs `rq0_0205`
- difficulty: `hard`
- human z: 0.3212 vs 0.5387; gold: `rq0_0205`
- RF: score_a=0.4444, score_b=0.4290, margin=0.0154, pred=`rq0_0074`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0205`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0074.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0205.png`

Code A excerpt:
```java
      
      out.writeObject(device.getDriver().getClass().getName());
      out.writeObject(device.getName());
      
      device.writeData(out);

```
Code B excerpt:
```java
/**
	 * Constructs a Oracle8iDialect
	 */
	public Oracle8iDialect() {
		super();
		registerCharacterTypeMappings();
		registerNumericTypeMappings();
		registerDateTimeTypeMappings();
		registerLargeObjectTypeMappings();
		registerReverseHibernateTypeMappings();
		registerFunctions();
		registerDefaultProperties();
	}
```

## RF_wrong__VLM_wrong
### full_rq0_seed42__rq0_0075__rq0_0257
- snippets: `rq0_0075` vs `rq0_0257`
- difficulty: `easy`
- human z: -0.8903 vs 0.3572; gold: `rq0_0257`
- RF: score_a=-0.7107, score_b=-0.7125, margin=0.0018, pred=`rq0_0075`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0075`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0075.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0257.png`

Code A excerpt:
```java
        
        File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome()  + File.separator + 
                "CaptureDevices" + File.separator + mCount + ".dat");
        
        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));
        
        dev.writeData(stream);

```
Code B excerpt:
```java
private void initOrdinaryPropertyPaths(Mapping mapping) throws MappingException {
		for ( int i = 0; i < getSubclassPropertyNameClosure().length; i++ ) {
			propertyMapping.initPropertyPaths( getSubclassPropertyNameClosure()[i],
					getSubclassPropertyTypeClosure()[i],
					getSubclassPropertyColumnNameClosure()[i],
					getSubclassPropertyColumnReaderClosure()[i],
					getSubclassPropertyColumnReaderTemplateClosure()[i],
					getSubclassPropertyFormulaTemplateClosure()[i],
					mapping );
		}
	}
```

## RF_wrong__VLM_invalid
### full_rq0_seed42__rq0_0021__rq0_0308
- snippets: `rq0_0021` vs `rq0_0308`
- difficulty: `easy`
- human z: 1.0323 vs -0.9134; gold: `rq0_0021`
- RF: score_a=-0.0492, score_b=-0.0488, margin=-0.0003, pred=`rq0_0308`, correct=False
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
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

